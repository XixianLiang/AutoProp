import typing
import collections
from transitions import Machine, State

DELETE = "Delete"
START = "start"
END = "end"


class DMF(object):

    def __init__(self) -> None:
        self.start_child_count:int = None
        self.start_state:str = None
        self.end_state:str = None
        self.keyword = None
        self.event_trace:list = []
        self.start_index:int = None
        self.keyword_index:int = None
        self.end_index:int = None

    def reset(self):
        self.start_child_count:int = None
        self.start_state:str = None
        self.end_state:str = None
        self.keyword = None
        self.event_trace:list = []
        self.start_index:int = None
        self.keyword_index:int = None
        self.end_index:int = None
    
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
        if state == "start":
            self.start_child_count = e["current_child_count"]
            self.start_index = i
        elif state in ["delete"]:
            self.keyword = e["keyword"]
            self.keyword_index = i
        elif state == "end":
            self.end_index = i
    
    def update_trace(self, event_cache):
        new_trace_len = self.end_index - self.start_index + 1
        if not self.event_trace or new_trace_len < (old_trace_len := len(self.event_trace)):
            self.event_trace = event_cache[self.start_index : self.end_index + 1]

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

        dmf_dict = dict()
        dmf:typing.Union[None, DMF]

        for i, e in enumerate(self):
            dmfID = e["dmfID"]
            current_child_count = e["current_child_count"]
            keyword = e["keyword"]
            
            if i == 71:
                pass

            if dmfID is None and keyword is None:
                continue

            if dmfID is not None:
                # 当该dmfID已经被初始化过，看看要不要覆盖（child是一样的）或者检查DMF（child不一致）
                if (dmf := dmf_dict.get(dmfID)):
                    dmf:DMF
                    assert dmf is dmf_dict[dmfID]
                    assert isinstance(dmf, DMF)
                    history_child_count = dmf.start_child_count
                    if history_child_count == current_child_count:
                        dmf.reset()
                        dmf.set_index("start", i, e)
                        continue
                    elif history_child_count != current_child_count:
                        if "Delete" == dmf.keyword and current_child_count == history_child_count - 1:
                            dmf.set_index("end", i, e)
                            dmf.update_trace(self)
                else:
                    dmf_dict[dmfID] = DMF()
                    dmf_dict[dmfID].set_index("start", i, e)
            
            if keyword is not None:
                for _, dmf in dmf_dict.items():
                    if dmf.start_index is not None:
                        dmf.set_index("delete", i, e)
        
        res:dict[str, DMF] = dict()
        for _dmfID, _dmf in dmf_dict.items():
            _dmf:DMF
            if _dmf.event_trace:
                res[_dmfID] = _dmf
        return res

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
    with open("cached_events_archived.txt", "r") as fp:
        for line in fp:
            cache.append(json.loads(line))
    return cache

if __name__ == "__main__":
    # sm = StateMachine()
    # print(sm.state)  # 输出: idle
    # sm.trigger("reach_dmf")  # 触发事件，进入 dmf_pre 状态
    # print(sm.state)  # 输出: dmf_pre
    cache = get_cache()
    d = cache.get_dmf_dict()
    d