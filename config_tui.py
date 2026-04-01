import json
import os
import sys
import importlib
from typing import Dict, Any, List, Optional, Type, Tuple
from pydantic import BaseModel

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, ListView, ListItem, Label, DataTable, Input, Static, Button
    from textual.containers import Container, Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.binding import Binding
    from textual import on
except ImportError:
    print("Please install textual: pip install textual")
    sys.exit(1)

# 确保能找到根目录
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ConfigLogic:
    def __init__(self):
        self.config_path = "config.json"
        self.config_data = self.load_config()
        self.ai_models = self.load_ai_models()
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

    def load_ai_models(self) -> Dict[str, Type[BaseModel]]:
        """加载所有 AI 模块的配置模型"""
        models = {}
        mapping = {
            "gemini": "ai_module.mod_gemini",
            "ollama": "ai_module.mod_ollama",
            "openai": "ai_module.mod_openai"
        }
        for key, path in mapping.items():
            try:
                mod = importlib.import_module(path)
                models[key] = mod.Module.get_config_model()
            except:
                pass
        return models

    def discover_modules(self) -> List[Dict]:
        """自动扫描目录发现配置模块"""
        directories = {
            "Shared AI": ("ai_module", "_shared_ai"),
            "Database": ("db", "_shared_db"),
            "Backends": ("backend", "")
        }
        
        results = []
        for category, (dir_name, ns_prefix) in directories.items():
            dir_path = os.path.join(project_root, dir_name)
            if not os.path.exists(dir_path): continue
            
            for filename in os.listdir(dir_path):
                if not filename.endswith(".py") or filename.startswith("__") or filename.endswith("_base.py") or filename.startswith("base_"):
                    continue
                
                mod_name = filename[:-3]
                import_path = f"{dir_name}.{mod_name}"
                
                try:
                    mod = importlib.import_module(import_path)
                    target_cls = None
                    
                    # 尝试发现合规的类
                    potential_names = ["Module", "DB", "ScannerBackend", "AssetBuilder", "OCRBackend", "AIDJRag"]
                    for name in potential_names:
                        attr = getattr(mod, name, None)
                        if attr and hasattr(attr, "get_config_model") and attr.get_config_model():
                            target_cls = attr
                            break
                    
                    if not target_cls:
                        for attr_name in dir(mod):
                            attr = getattr(mod, attr_name)
                            if isinstance(attr, type) and hasattr(attr, "get_config_model") and attr.get_config_model():
                                if attr.__module__ != "mod_base":
                                    target_cls = attr
                                    break
                    
                    if target_cls:
                        is_backend = category == "Backends"
                        if is_backend:
                            ns = mod_name
                        else:
                            ns = getattr(target_cls, "shared_namespace", None)
                            if not ns:
                                clean_name = mod_name[4:] if mod_name.startswith("mod_") else mod_name
                                ns = f"{ns_prefix}_{clean_name}"

                        results.append({
                            "category": category,
                            "name": mod_name,
                            "ns": ns,
                            "model": target_cls.get_config_model(),
                            "is_backend": is_backend
                        })
                except:
                    pass
        
        results.sort(key=lambda x: (x['category'], x['name']))
        return results

class EditValueModal(ModalScreen[str]):
    def __init__(self, field_name: str, current_value: Any):
        super().__init__()
        self.field_name = field_name
        self.current_value = current_value

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Editing: [bold cyan]{self.field_name}[/bold cyan]")
            yield Input(value=str(self.current_value), id="field_input")
            with Horizontal(id="buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", variant="error", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.dismiss(self.query_one("#field_input").value)
        else:
            self.dismiss(None)

class ConfigApp(App):
    CSS = """
    Screen { background: $surface; }
    #main_container { layout: horizontal; }
    #module_list { width: 30%; border-right: tall $primary; background: $surface; }
    #detail_panel { width: 70%; padding: 1; }
    #dialog { padding: 1 2; background: $panel; border: thick $primary; width: 60; height: auto; align: center middle; }
    #buttons { margin-top: 1; align: center middle; }
    DataTable { height: 1fr; border: solid $primary; }
    ListItem { padding: 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("s", "save_config", "Save All", show=True),
        Binding("o", "toggle_override", "Toggle Override", show=True),
        Binding("r", "reset_item", "Reset Item", show=True),
    ]

    def __init__(self):
        super().__init__()
        self.logic = ConfigLogic()
        self.selected_module = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main_container"):
            yield ListView(id="module_list")
            with Vertical(id="detail_panel"):
                yield Label("Module Configuration", id="detail_title")
                yield Static("", id="override_status")
                yield DataTable(id="config_table")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_module_list()
        self.query_one("#config_table").add_columns("Field", "Value", "Description", "Default")
        self.query_one("#config_table").cursor_type = "row"

    def refresh_module_list(self):
        lv = self.query_one("#module_list")
        lv.clear()
        for m in self.logic.registry:
            status = "●" if m['ns'] in self.logic.config_data else "○"
            lv.append(ListItem(Label(f" {status} {m['name']} ({m['ns']})"), name=m['ns']))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        ns = event.item.name
        self.selected_module = next(m for m in self.logic.registry if m['ns'] == ns)
        self.refresh_config_table()

    def refresh_config_table(self):
        if not self.selected_module: return
        
        from rich.text import Text
        ns = self.selected_module['ns']
        model = self.selected_module['model']
        table = self.query_one("#config_table")
        table.clear()
        
        if ns not in self.logic.config_data:
            self.logic.config_data[ns] = {}

        # 1. 显示 Backend 自身的逻辑字段
        fields = model.model_fields
        shared_ns = "_shared_backend" if self.selected_module.get("is_backend") else None
        
        for f_name, f_info in fields.items():
            if f_name in self.logic.config_data[ns]:
                val, style = str(self.logic.config_data[ns][f_name]), "bold green"
            elif shared_ns and f_name in self.logic.config_data.get(shared_ns, {}):
                val, style = str(self.logic.config_data[shared_ns][f_name]), "yellow"
            else:
                val, style = str(f_info.default), "dim"
            table.add_row(f_name, Text(val, style=style), f_info.description or "", str(f_info.default), key=f"base:{f_name}")

        # 2. 如果是 Backend，检测其 AI 穿透配置
        if self.selected_module.get("is_backend"):
            ai_proxies = [("", ns)]
            if self.selected_module['name'] in ("asset_builder", "aidj_rag"):
                ai_proxies.append(("ollama_embed", "ollama_embed"))

            for sub_key, proxy_ns in ai_proxies:
                self._add_ai_proxy_rows(table, ns, sub_key, proxy_ns)

            # 3. 检测其 DB 穿透配置 (postgresql)
            if self.selected_module['name'] in ("asset_builder", "aidj_rag"):
                db_ns = "db_pg"
                db_reg = next((m for m in self.logic.registry if m['ns'].endswith(db_ns)), None)
                if db_reg:
                    db_model = db_reg['model']
                    shared_db_ns = db_reg['ns']
                    if db_ns not in self.logic.config_data[ns]:
                        self.logic.config_data[ns][db_ns] = {}
                    
                    table.add_row(f"[bold magenta]--- DB: {db_ns} ---[/]", "", "", "", key=f"divider:{db_ns}")
                    
                    sub_ov_val = self.logic.config_data[ns][db_ns].get("_override", False)
                    sub_ov_style = "bold green" if "_override" in self.logic.config_data[ns][db_ns] else "dim"
                    table.add_row(f"  [magenta]{db_ns}._override[/]", Text(str(sub_ov_val), style=sub_ov_style), "Override shared DB settings", "False", key=f"sub:{db_ns}:_override")

                    for f_name, f_info in db_model.model_fields.items():
                        if f_name in self.logic.config_data[ns][db_ns]:
                            val, style = str(self.logic.config_data[ns][db_ns][f_name]), "bold green"
                        elif f_name in self.logic.config_data.get(shared_db_ns, {}):
                            val, style = str(self.logic.config_data[shared_db_ns][f_name]), "yellow"
                        else:
                            val, style = str(f_info.default), "dim"
                        table.add_row(f"  [magenta]{db_ns}.{f_name}[/]", Text(val, style=style), f_info.description or "", str(f_info.default), key=f"sub:{db_ns}:{f_name}")
        
        override = self.logic.config_data[ns].get("_override", False)
        self.query_one("#override_status").update(f"Override Shared Backend: [bold {'green' if override else 'red'}]{override}[/]")
        self.query_one("#detail_title").update(f"Configuring: [bold cyan]{self.selected_module['name']}[/]")

    def _add_ai_proxy_rows(self, table, ns, sub_key, display_name):
        """展示 AI 代理 (mod_general) 的穿透配置"""
        from rich.text import Text
        target_dict = self.logic.config_data[ns]
        if sub_key:
            if sub_key not in target_dict: target_dict[sub_key] = {}
            target_dict = target_dict[sub_key]
        
        ai_type = target_dict.get("ai") or self.logic.config_data.get("_shared_ai_general", {}).get("ai", "gemini")
        ai_style = "bold green" if "ai" in target_dict else "dim"
        
        key_prefix = f"sub:{sub_key}:" if sub_key else "sub::"
        display_prefix = f"{sub_key}." if sub_key else ""

        table.add_row(f"[bold yellow]--- AI: {display_name} ---[/]", "", "", "", key=f"divider:{display_name}")
        table.add_row(f"  [yellow]{display_prefix}ai[/]", Text(ai_type, style=ai_style), f"Select AI backend for {display_name}", "gemini", key=f"{key_prefix}ai")

        if ai_type in self.logic.ai_models:
            ai_model = self.logic.ai_models[ai_type]
            shared_ai_ns = f"_shared_ai_{ai_type}"
            if ai_type not in target_dict: target_dict[ai_type] = {}
            
            sub_ov_val = target_dict[ai_type].get("_override", False)
            sub_ov_style = "bold green" if "_override" in target_dict[ai_type] else "dim"
            table.add_row(f"  [cyan]{display_prefix}{ai_type}._override[/]", Text(str(sub_ov_val), style=sub_ov_style), "Override shared AI settings", "False", key=f"{key_prefix}{ai_type}:_override")

            for f_name, f_info in ai_model.model_fields.items():
                if f_name in target_dict[ai_type]:
                    val, style = str(target_dict[ai_type][f_name]), "bold green"
                elif f_name in self.logic.config_data.get(shared_ai_ns, {}):
                    val, style = str(self.logic.config_data[shared_ai_ns][f_name]), "yellow"
                else:
                    val, style = str(f_info.default), "dim"
                table.add_row(f"  [cyan]{display_prefix}{ai_type}.{f_name}[/]", Text(val, style=style), f_info.description or "", str(f_info.default), key=f"{key_prefix}{ai_type}:{f_name}")

    @on(DataTable.RowSelected)
    def handle_row_selection(self, event: DataTable.RowSelected):
        key_parts = event.row_key.value.split(":")
        if key_parts[0] == "divider": return
        
        row_type = key_parts[0] # 'base' or 'sub'
        ns = self.selected_module['ns']
        
        if row_type == "base":
            field_name = key_parts[1]
            current_val = self.logic.config_data[ns].get(field_name, "")
            model = self.selected_module['model']
            f_info = model.model_fields.get(field_name) if field_name != "ai" else None
            
            def check_result(new_value: str):
                if new_value is None: return
                self._update_val(self.logic.config_data[ns], field_name, f_info, new_value)
                self.refresh_config_table()
                self.refresh_module_list()
            self.push_screen(EditValueModal(field_name, current_val), check_result)
        else:
            sub_key = key_parts[1]
            target_dict = self.logic.config_data[ns]
            if sub_key: target_dict = target_dict[sub_key]
            field_path = key_parts[2:]
            
            if len(field_path) == 1: 
                field_name = field_path[0]
                current_val = target_dict.get(field_name, "")
                if field_name != "ai":
                    reg_item = next((m for m in self.logic.registry if m['ns'].endswith(sub_key)), None)
                    model = reg_item['model'] if reg_item else None
                    f_info = model.model_fields.get(field_name) if model and field_name != "_override" else None
                else:
                    f_info = None
            else:
                ai_type, field_name = field_path[0], field_path[1]
                current_val = target_dict[ai_type].get(field_name, "")
                model = self.logic.ai_models.get(ai_type)
                if not model:
                    reg_item = next((m for m in self.logic.registry if m['ns'].endswith(ai_type)), None)
                    if reg_item: model = reg_item['model']
                f_info = model.model_fields.get(field_name) if model and field_name != "_override" else None
                target_dict = target_dict[ai_type]

            def check_result(new_value: str):
                if new_value is None: return
                self._update_val(target_dict, field_name, f_info, new_value)
                self.refresh_config_table()
                self.refresh_module_list()
            self.push_screen(EditValueModal(field_name, current_val), check_result)

    def _update_val(self, target_dict, field_name, f_info, new_value):
        try:
            if field_name == "ai" or field_name == "_override":
                if field_name == "_override":
                    target_dict[field_name] = new_value.lower() in ("true", "1", "yes")
                else:
                    target_dict[field_name] = new_value
            elif f_info:
                if f_info.annotation == int: target_dict[field_name] = int(new_value)
                elif f_info.annotation == bool: target_dict[field_name] = new_value.lower() in ("true", "1", "yes")
                elif f_info.annotation == float: target_dict[field_name] = float(new_value)
                elif f_info.annotation == list: target_dict[field_name] = [v.strip() for v in new_value.split(",")]
                else: target_dict[field_name] = new_value
        except:
            self.notify("Invalid Type", severity="error")

    def action_reset_item(self):
        """重置当前选中的配置项"""
        table = self.query_one("#config_table")
        if table.cursor_row is None: return
        row_keys = list(table.rows.keys())
        row_key = row_keys[table.cursor_row].value
        key_parts = row_key.split(":")
        if key_parts[0] == "divider": return
        ns = self.selected_module['ns']
        if key_parts[0] == "base":
            field_name = key_parts[1]
            if field_name in self.logic.config_data[ns]: self.logic.config_data[ns].pop(field_name)
        else:
            sub_key = key_parts[1]
            target_dict = self.logic.config_data[ns]
            if sub_key: target_dict = target_dict[sub_key]
            field_path = key_parts[2:]
            if len(field_path) == 1:
                field_name = field_path[0]
                if field_name in target_dict: target_dict.pop(field_name)
            else:
                ai_type, field_name = field_path[0], field_path[1]
                if ai_type in target_dict and field_name in target_dict[ai_type]:
                    target_dict[ai_type].pop(field_name)
        self.refresh_config_table()
        self.notify("Item reset to inherited/default value")

    def action_toggle_override(self):
        if self.selected_module:
            ns = self.selected_module['ns']
            current = self.logic.config_data[ns].get("_override", False)
            self.logic.config_data[ns]["_override"] = not current
            self.refresh_config_table()

    def action_save_config(self):
        self.logic.save_config()
        self.notify("Configuration Saved!", severity="information")

if __name__ == "__main__":
    app = ConfigApp()
    app.run()
