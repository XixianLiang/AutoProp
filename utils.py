import typing
import collections
import copy
from transitions import Machine, State

DELETE = "Delete"
SEARCH = "Search"
ADD = "Add"
START = "start"
END = "end"


def case_insensitive_check(self:str, others:typing.Union[str, list[str]]):
    if not isinstance(self, str) or not (isinstance(others, str) or isinstance(others, list)):
        raise AttributeError 
    if isinstance(others, str):
        return self.lower() == others.lower()
    elif isinstance(others, list):
        return 

class DMF_dict(dict):
    """
    用来存储单个DMF的dict，key为 Add, Delete, Search, Update。
    value 为 DMF
    """

    def __setitem__(self, key: str, value: "DMF_extractor") -> None:
        # 更新 Search 的关键字，对Value进行清洗（强转为DMF类）
        key = SEARCH if SEARCH in key else key
        value = DMF(value) if isinstance(value, DMF_extractor) else value

        if key not in (dmf_keyword := [SEARCH, ADD, DELETE]):
            raise AttributeError(f"Invalid DMF_dict key. key should be in {dmf_keyword}.")
        
        # 如果当前的event trace比已知的event trace小，才更新。
        if not (old_value := self.get(key)):
            return super().__setitem__(key, value)
        elif old_value and len(old_value.event_trace) > len(value.event_trace):
            return super().__setitem__(key, value)
        else:
            return 

class DMF(dict):
    """
    用来存储DMF结果的类，比DMF_extractor更精简
    """
    def __init__(self, _dmf:"DMF_extractor"):
        self.event_trace = list(_dmf.event_trace)
        self.start_text_lt = list(_dmf.start_text_lt)
        self.end_text_lt = list(_dmf.end_text_lt)
        self.start_state = _dmf.start_state
        self.changed_item = self.get_changed_item(_dmf)
        self.init_dict()

    def init_dict(self):
        self["event_trace"] = self.event_trace
        self["start_text_lt"] = self.start_text_lt
        self["end_text_lt"] = self.end_text_lt
        self["start_state"] = self.start_state

    def get_changed_item(self, _dmf:"DMF_extractor"):
        if _dmf.keyword == ADD:
            return (_dmf.end_text_lt - _dmf.start_text_lt)[0]
        elif _dmf.keyword == DELETE:
            return (_dmf.start_text_lt - _dmf.end_text_lt)[0]
        


class DMF_extractor:
    """
    用来抽取DMF时用的类
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.start_child_count:int = None
        self.start_state:str = None
        self.end_state:str = None
        self.keyword:str = None
        self.event_trace:list = []
        self.start_index:int = None
        self.keyword_index:int = None
        self.end_index:int = None
        self.start_text_lt = None
        self.end_text_lt = None
    
    @property
    def trace_len(self):
        return len(self.event_trace) if self.event_trace else -1
    
    def to_dict(self):
        return {"start_child_count":self.start_child_count,
                "start_state":self.start_state,
                "end_state":self.end_state,
                "keyword":self.keyword,
                "event_trace":self.event_trace}

    def set_index(self, state, i, e):
        if self.end_index:
            return
        if state == "start":
            self.start_child_count = e["current_child_count"]
            self.start_index = i
            self.start_text_lt = subtractList(e["text_lt"])
        elif state in [DELETE, ADD] or SEARCH in state:
            self.keyword = e["keyword"]
            self.keyword_index = i
        elif state == "end":
            self.end_index = i
            self.end_text_lt = subtractList(e["text_lt"])
    
    def update_trace(self, event_cache):
        new_trace_len = self.end_index - self.start_index + 1
        if not self.event_trace or new_trace_len < (old_trace_len := len(self.event_trace)):
            self.event_trace = event_cache[self.start_index : self.end_index + 1]


class subtractList(collections.UserList):
    
    def __sub__(self, others):
        res = copy.deepcopy(self)
        for item in others:
            if item in res:
                res.remove(item)
        return res

class Queue(collections.UserList):
    def __init__(self, args: typing.Optional[typing.Iterable] = None, max_size: int = 100):
        self._max_size = max_size
        if args is not None:
            super().__init__(args)
        else:
            super().__init__()

    def append(self, object: typing.Any) -> None:
        if len(self) == self._max_size:
            del self[0]
        super().append(object)

    def __str__(self) -> str:
        return f"Queue({super().__str__()})"
    
    def get_dmf_dict(self):

        dmf_dict:dict[str, DMF_extractor] = dict()
        dmf:typing.Union[None, DMF_extractor]

        for i, e in enumerate(self):
            dmfID = e["dmfID"]
            current_child_count = e["current_child_count"]
            text_lt = subtractList(e["text_lt"])
            keyword = e["keyword"]

            if dmfID is None and keyword is None:
                continue

            if dmfID is not None:
                # 当该dmfID已经被初始化过，看看要不要覆盖（child是一样的）或者检查DMF（child不一致）

                if not (dmf := dmf_dict.get(dmfID)):
                    # 初始化 DMF
                    dmf = DMF_extractor()
                    dmf_dict[dmfID] = dmf
                    dmf_dict[dmfID].set_index("start", i, e)
                    

                dmf:DMF_extractor
                assert dmf is dmf_dict[dmfID]
                assert isinstance(dmf, DMF_extractor)
                history_child_count = dmf.start_child_count
                history_text_lt = dmf.start_text_lt
                if history_child_count == current_child_count:
                    if history_text_lt == text_lt and dmf.keyword and SEARCH not in dmf.keyword:
                        dmf.reset()
                        dmf.set_index("start", i, e)
                # 不同的时候进行处理，这里处理的是add和delete
                elif history_child_count != current_child_count:
                    assert isinstance(text_lt, subtractList)
                    assert isinstance(history_text_lt, subtractList)
                    # if "Delete" == dmf.keyword and current_child_count == history_child_count - 1:
                    if dmf.keyword and DELETE == dmf.keyword and len(history_text_lt - text_lt) == 1 and current_child_count == history_child_count - 1:
                        dmf.set_index("end", i, e)
                        dmf.update_trace(self)
                    elif dmf.keyword and ADD == dmf.keyword and len(text_lt - history_text_lt) == 1 and current_child_count == history_child_count + 1:
                        dmf.set_index("end", i, e)
                        dmf.update_trace(self)

                        # 设置 keyword
            if keyword is not None:
                for _, dmf in dmf_dict.items():
                    if dmf.start_index is not None:
                        dmf.set_index(keyword, i, e)
            
        
            # 处理search
            for _dmfID, _dmf in dmf_dict.items():
                if _dmf.keyword and SEARCH in _dmf.keyword:
                    target_text = _dmf.keyword.split("_", 1)[-1]
                    if text_lt and all(target_text in _ for _ in text_lt) \
                        and _dmf.start_text_lt and any(target_text in _ for _ in _dmf.start_text_lt):
                        _dmf.set_index("end", i, e)
                        _dmf.update_trace(self)
        
        res:dict[str, DMF_extractor] = dict()
        for _dmfID, _dmf in dmf_dict.items():
            _dmf:DMF_extractor
            if _dmf.event_trace:
                res[_dmfID] = _dmf
        return res

# class DMF_extractor:

#     def __init__(self) -> None:
#         self.reset()

#     def reset(self):
#         self.start_child_count:int = None
#         self.start_state:str = None
#         self.end_state:str = None
#         self.keyword:list[str] = []
#         self.event_trace:list = []
#         self.start_index:int = None
#         self.keyword_index:int = None
#         self.end_index:int = None
#         self.start_text_lt = None
#         self.end_text_lt = None
    
#     @property
#     def trace_len(self):
#         return len(self.event_trace) if self.event_trace else -1
    
#     def to_dict(self):
#         return {"start_child_count":self.start_child_count,
#                 "start_state":self.start_state,
#                 "end_state":self.end_state,
#                 "keyword":self.keyword,
#                 "event_trace":self.event_trace}

#     def set_index(self, state, i, e):
#         if state == "start":
#             self.start_child_count = e["current_child_count"]
#             self.start_index = i
#             self.start_text_lt = subtractList(e["text_lt"])
#         elif state in [DELETE, ADD] or SEARCH in state:
#             self.keyword.append(e["keyword"])
#             self.keyword_index = i
#         elif state == "end":
#             self.end_index = i
#             self.end_text_lt = subtractList(e["text_lt"])
    
#     def update_trace(self, event_cache):
#         new_trace_len = self.end_index - self.start_index + 1
#         if not self.event_trace or new_trace_len < (old_trace_len := len(self.event_trace)):
#             self.event_trace = event_cache[self.start_index : self.end_index + 1]

# class subtractList(collections.UserList):
    
#     def __sub__(self, others):
#         res = copy.deepcopy(self)
#         for item in others:
#             if item in res:
#                 res.remove(item)
#         return res

# class Queue(collections.UserList):
#     def __init__(self, args: typing.Optional[typing.Iterable] = None, max_size: int = 100):
#         self._max_size = max_size
#         if args is not None:
#             super().__init__(args)
#         else:
#             super().__init__()

#     def append(self, object: typing.Any) -> None:
#         if len(self) == self._max_size:
#             del self[0]
#         super().append(object)

#     def __str__(self) -> str:
#         return f"Queue({super().__str__()})"
    
#     def get_dmf_dict(self):

#         dmf_dict:dict[str, DMF_extractor] = dict()
#         dmf:typing.Union[None, DMF_extractor]

#         for i, e in enumerate(self):
#             dmfID = e["dmfID"]
#             current_child_count = e["current_child_count"]
#             text_lt = subtractList(e["text_lt"])
#             keyword = e["keyword"]

#             if i == 4:
#                 pass

#             if dmfID is None and keyword is None:
#                 continue

#             if dmfID is not None:
#                 # 当该dmfID已经被初始化过，看看要不要覆盖（child是一样的）或者检查DMF（child不一致）

#                 if not (dmf := dmf_dict.get(dmfID)):
#                     # 初始化 DMF
#                     dmf = DMF_extractor()
#                     dmf_dict[dmfID] = dmf
#                     dmf_dict[dmfID].set_index("start", i, e)
                    

#                 dmf:DMF_extractor
#                 assert dmf is dmf_dict[dmfID]
#                 assert isinstance(dmf, DMF_extractor)
#                 history_child_count = dmf.start_child_count
#                 history_text_lt = dmf.start_text_lt
#                 if history_child_count == current_child_count:
#                     if history_text_lt == text_lt and SEARCH not in dmf.keyword:
#                         dmf.reset()
#                         dmf.set_index("start", i, e)
#                 # 不同的时候进行处理，这里处理的是add和delete
#                 elif history_child_count != current_child_count:
#                     assert isinstance(text_lt, subtractList)
#                     assert isinstance(history_text_lt, subtractList)
#                     # if "Delete" == dmf.keyword and current_child_count == history_child_count - 1:
#                     if DELETE in dmf.keyword and len(history_text_lt - text_lt) == 1 and current_child_count == history_child_count - 1:
#                         dmf.set_index("end", i, e)
#                         dmf.update_trace(self)
#                     elif ADD in dmf.keyword and len(text_lt - history_text_lt) == 1 and current_child_count == history_child_count + 1:
#                         dmf.set_index("end", i, e)
#                         dmf.update_trace(self)
            
#                         # 设置 keyword
#             if keyword is not None:
#                 for _, dmf in dmf_dict.items():
#                     if dmf.start_index is not None:
#                         dmf.set_index(keyword, i, e)
            
        
#             # 处理search
#             for _dmfID, _dmf in dmf_dict.items():
#                 for search_key in _dmf.keyword:
#                     if SEARCH in search_key:
#                         target_text = search_key.split("_", 1)[-1]
#                         if text_lt is not None and all(target_text in _ for _ in text_lt):
#                             _dmf.set_index("end", i, e)
#                             _dmf.update_trace(self)
            
#             if keyword == ADD:
#                 pass
        
#         res:dict[str, DMF_extractor] = dict()
#         for _dmfID, _dmf in dmf_dict.items():
#             _dmf:DMF_extractor
#             if _dmf.event_trace:
#                 res[_dmfID] = _dmf
#         return res

class Stack(collections.UserList):
    def push(self, item):
        self.append(item)
    
    def pop(self):
        super().pop(i=-1)

class Model:
    def test(self):
        print("Entering dmf_pre state")

class StateMachine:
    states = [
        {"name": "idle"},
        State(name="dmf_pre", on_enter=["test"]),
        {"name": "dmf_delete"},
        {"name": "dmf_post"}
    ]
    
    transitions = [
        {"trigger": "reach_dmf", "source": "idle", "dest": "dmf_pre"},
        {"trigger": "reach_dmf", "source": "dmf_pre", "dest": "dmf_pre"},
        {"trigger": "reach_dmf", "source": "dmf_delete", "dest": "dmf_post"},
        {"trigger": "reset", "source": "dmf_post", "dest": "idle"}
    ]

    def test(self):
        print(f"entered state")

    def __init__(self):
        self._m = None
        self._q = Queue()
        self.model = Model()  # 创建 Model 的实例
        self.init()

    def init(self):
        self._m = Machine(states=self.states, transitions=self.transitions, initial="idle")
    
    def trigger(self, event: str):
        return self._m.trigger(event)
    
    @property
    def state(self):
        return self._m.state

def get_cache():
    import json
    cache = Queue()
    with open("cached_events.txt", "r") as fp:
        for line in fp:
            cache.append(json.loads(line))
    return cache



def get_content(obj, key):
    if isinstance(obj[key], bool):
        return "true" if obj[key] else "false"
    return obj[key] if obj[key] else ""

def generate_xml_node(o):
    xml_node = f"""<node text="{get_content(o, "text")}"\
                resource-id="{get_content(o, "resource_id")}"\
                class="{get_content(o, "class")}"\
                package="{get_content(o, "package")}"\
                content-desc="{get_content(o, "content_description")}"\
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

if __name__ == "__main__":
    # sm = StateMachine()
    # print(sm.state)  # 输出: idle
    # sm.trigger("reach_dmf")  # 触发事件，进入 dmf_pre 状态
    # print(sm.state)  # 输出: dmf_pre
    # cache = get_cache()
    # d = cache.get_dmf_dict()
    # d
    import json
    dmf_dict:dict[str, DMF_dict] = dict()
    with open("cached_events.txt", "r") as fp:
        q = Queue([json.loads(_) for _ in fp])
        res = q.get_dmf_dict()
        res

        for _dmfID, _dmf in res.items():
            dmf_dict[_dmfID] = dmf_dict.get(_dmfID, DMF_dict())
            dmf_dict[_dmfID][_dmf.keyword] = dmf_dict[_dmfID].get(_dmf.keyword, _dmf)

        
    with open("output_dmf.json", "w") as fp:
        json.dump(dmf_dict, fp)