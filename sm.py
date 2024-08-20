from transitions import Machine, State
from queue import Queue

class Model:
    def test(self):
        print("Entering dmf_pre state")

class StateMachine:
    states = [
        {"name": "idle"},
        {"name": "dmf_pre", "on_enter":"test"},
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
        self._m = Model()
        self._q = Queue() # 创建 Model 的实例
        self.init()

    def init(self):
        Machine(model=self._m, states=self.states, transitions=self.transitions, initial="idle")
    
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