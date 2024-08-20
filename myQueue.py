import typing
import collections
from transitions import Machine, State

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
    
    def dmf_record(self):
        pass

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

if __name__ == "__main__":
    sm = StateMachine()
    print(sm.state)  # 输出: idle
    sm.trigger("reach_dmf")  # 触发事件，进入 dmf_pre 状态
    print(sm.state)  # 输出: dmf_pre
