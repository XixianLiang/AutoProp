import json

a = '{"package": "org.tasks", "visible": true, "checkable": false, "child_count": 0, "editable": false, "clickable": true, "is_password": false, "focusable": true, "enabled": true, "content_description": "Create new task", "children": [], "focused": false, "bounds": [[456, 1524], [624, 1692]], "resource_id": "org.tasks:id/fab", "checked": false, "text": null, "class": "android.widget.ImageButton", "scrollable": false, "selected": false, "long_clickable": false, "parent": 7, "temp_id": 32, "size": "168*168", "signature": "[class]android.widget.ImageButton[resource_id]org.tasks:id/fab[description]Create new task[text]None[enabled,,,visible]", "view_str": "f97b1435f54a2163a6782cd32874bac7", "content_free_signature": "[class]android.widget.ImageButton[resource_id]org.tasks:id/fab", "allowed_actions": ["touch"], "special_attrs": [], "local_id": "7", "desc": "<button text=\'Create new task\' bound_box=456,1524,624,1692></button>"}'
o = json.loads(a)
o
"""
<node index="2" text="" resource-id="" class="android.widget.ImageView" package="org.tasks" content-desc="More options" checkable="false" checked="false" clickable="true" enabled="true" focusable="true" focused="false" scrollable="false" long-clickable="true" password="false" selected="false" visible-to-user="true" bounds="[975,1657][1080,1783]" />
"""

def get_content(obj, key):
    if isinstance(obj[key], bool):
        return "true" if obj[key] else "false"
    return obj[key] if obj[key] else ""

def generate_xml_node(o):
    xml_node = f"""<node text="{get_content(o, "text")}"\
                resourceId="{get_content(o, "resource_id")}"\
                class="{get_content(o, "class")}"\
                package="{get_content(o, "package")}"\
                content-descrip="{get_content(o, "content_description")}"\
                checkable="{get_content(o, "checkable")}"\
                checked="{get_content(o, "checked")}"\
                clickable="{get_content(o, "clickable")}"\
                enabled="{get_content(o, "enabled")}"\
                focusable="{get_content(o, "focusable")}"\
                focused="{get_content(o, "focused")}"\
                scrollable="{get_content(o, "scrollable")}"\
                long-clickable="{get_content(o, "long_clickable")}"\
                selected="{get_content(o, "selected")}"\
                visible-to-user="{get_content(o, "visible")}"\
                bounds="{o["bounds"][0]}{o["bounds"][1]}"\
                {'index="{}"'.format(o["index"]) if hasattr(o, "index") else ""}\
                />"""
    
    return xml_node