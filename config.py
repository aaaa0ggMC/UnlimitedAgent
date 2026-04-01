import json
import os
import sys
import importlib
from typing import Dict, Any, Type, List, Optional
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.live import Live
from rich import print as rprint

# 确保能找到根目录
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

console = Console()

class ConfigManager:
    def __init__(self):
        self.config_path = "config.json"
        self.config_data = self.load_config()
        self.registry = self.discover_modules()

    def load_config(self) -> Dict:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=2)
        console.print("[bold green]✔ Configuration saved![/bold green]")

    def discover_modules(self) -> Dict[str, Dict]:
        """扫描所有支持配置的模块"""
        registry = {
            "Shared AI": {
                "ai_gemini": ("ai_module.mod_gemini", "_shared_ai_gemini"),
                "ai_ollama": ("ai_module.mod_ollama", "_shared_ai_ollama"),
                "ai_openai": ("ai_module.mod_openai", "_shared_ai_openai"),
            },
            "Database": {
                "postgresql": ("db.postgresql", "_shared_db_pg"),
            },
            "Backends": {
                "scanning_gemini": ("backend.scanning_gemini", "scanning_gemini"),
                "asset_builder": ("backend.asset_builder", "asset_builder"),
                "gemini_ocr": ("backend.gemini_ocr", "gemini_ocr"),
            }
        }
        
        # 验证并加载类
        results = {}
        for category, modules in registry.items():
            results[category] = []
            for name, (path, ns) in modules.items():
                try:
                    mod = importlib.import_module(path)
                    cls = getattr(mod, "Module", None) or getattr(mod, "DB", None) or getattr(mod, "ScannerBackend", None) or getattr(mod, "AssetBuilder", None) or getattr(mod, "OCRBackend", None)
                    if cls and hasattr(cls, "get_config_model") and cls.get_config_model():
                        results[category].append({
                            "name": name,
                            "class": cls,
                            "ns": ns,
                            "model": cls.get_config_model()
                        })
                except Exception as e:
                    # console.print(f"[red]Failed to load {name}: {e}[/red]")
                    pass
        return results

    def show_main_menu(self):
        table = Table(title="UnlimitedAgent Modules", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=4)
        table.add_column("Category", style="cyan")
        table.add_column("Module Name", style="green")
        table.add_column("Namespace", style="yellow")
        table.add_column("Status", justify="center")

        flat_list = []
        idx = 1
        for category, modules in self.registry.items():
            for m in modules:
                status = "[green]Configured[/green]" if m['ns'] in self.config_data else "[red]Empty[/red]"
                table.add_row(str(idx), category, m['name'], m['ns'], status)
                flat_list.append(m)
                idx += 1

        console.print(table)
        choice = Prompt.ask("\nSelect a module ID to configure (or 'q' to quit, 's' to save)", default="q")
        
        if choice.lower() == 'q':
            sys.exit(0)
        if choice.lower() == 's':
            self.save_config()
            return

        try:
            m_idx = int(choice) - 1
            if 0 <= m_idx < len(flat_list):
                self.edit_module(flat_list[m_idx])
        except ValueError:
            console.print("[red]Invalid choice[/red]")

    def edit_module(self, module_info: Dict):
        ns = module_info['ns']
        model = module_info['model']
        name = module_info['name']
        
        if ns not in self.config_data:
            self.config_data[ns] = {}

        while True:
            table = Table(title=f"Configuration for [bold cyan]{name}[/bold cyan] ({ns})", show_header=True, header_style="bold blue")
            table.add_column("ID", style="dim", width=4)
            table.add_column("Field", style="yellow")
            table.add_column("Description", style="italic")
            table.add_column("Current Value", style="green")
            table.add_column("Default", style="dim")

            fields = model.model_fields
            idx_to_field = {}
            for i, (f_name, f_info) in enumerate(fields.items(), 1):
                cur_val = self.config_data[ns].get(f_name, "[dim italic]unset[/dim italic]")
                desc = f_info.description or f_name
                table.add_row(str(i), f_name, desc, str(cur_val), str(f_info.default))
                idx_to_field[i] = (f_name, f_info)

            console.clear()
            console.print(table)
            
            # 检查是否有 _override
            override_status = self.config_data[ns].get("_override", False)
            console.print(f"Override Shared Config: {'[green]Yes[/green]' if override_status else '[red]No[/red]'}")
            
            choice = Prompt.ask("\nSelect ID to edit, 'o' to toggle override, 'b' to back", default="b")
            
            if choice.lower() == 'b':
                break
            if choice.lower() == 'o':
                self.config_data[ns]["_override"] = not override_status
                continue

            try:
                f_idx = int(choice)
                if f_idx in idx_to_field:
                    f_name, f_info = idx_to_field[f_idx]
                    self.ask_and_set(ns, f_name, f_info)
            except ValueError:
                pass

    def ask_and_set(self, ns, field_name, field_info):
        current_val = self.config_data[ns].get(field_name, field_info.default)
        prompt_text = f"Enter value for {field_name}"
        
        if field_info.annotation == bool:
            val = Confirm.ask(prompt_text, default=bool(current_val))
        else:
            val = Prompt.ask(prompt_text, default=str(current_val) if current_val is not None else None)
            
            # 类型转换
            try:
                if field_info.annotation == int: val = int(val)
                elif field_info.annotation == float: val = float(val)
                elif field_info.annotation == list:
                    if isinstance(val, str): val = [v.strip() for v in val.split(",")]
            except:
                console.print("[red]Invalid type, value not changed[/red]")
                return

        self.config_data[ns][field_name] = val

    def run(self):
        console.clear()
        console.print(Panel.fit("🚀 [bold magenta]UnlimitedAgent Configuration Suite[/bold magenta]", border_style="magenta"))
        while True:
            self.show_main_menu()

if __name__ == "__main__":
    manager = ConfigManager()
    manager.run()
