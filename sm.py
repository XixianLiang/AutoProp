from transitions import Machine, State
from myQueue import Queue

class Model:
    def print_state(self):
        print()

class StateMachine:
    states = [
        {"name": "idle", "on_enter":"print_state"},
        {"name": "dmf_pre"},
        {"name": "dmf_delete"},
        {"name": "dmf_post"}
    ]
    
    transitions = [
        {"source": "idle", "dest": "dmf_pre", "trigger": "reach_dmf"},
        {"source": "dmf_pre", "dest": "dmf_delete", "trigger": "delete"},
        {"source": "dmf_delete", "dest": "dmf_post", "trigger": "reach_dmf"},
        {"source": "dmf_post", "dest": "idle", "trigger": "reset"}
    ]

    def __init__(self):
        self._m = Model()
        self._q = Queue() # 创建 Model 的实例
        self.init()

    def init(self):
        Machine(model=self._m, states=self.states, transitions=self.transitions, initial="idle")
    
    def trigger(self, transit: str):
        from_state = self.state
        result = self._m.trigger(transit)
        to_state = self.state
        print(f"transititon({from_state} -> {to_state})")
        return result
    
    def push(self, event):
        self._q.append(event)
    
    @property
    def state(self):
        return self._m.state

class SM_Controller():
    def __init__(self):
        self.sm = StateMachine()

    def push(self, event):
        if event["dmfID"] is not None:
            self.sm.trigger("reach_dmf")
            self.sm.push(event)
            return
        
        if event["keyword"] is not None:
            self.sm.trigger((keyword := event["keyword"]))
            self.sm.push(event)
            return

        if self.sm.state != "idle":
            self.sm.push(event)
            return


event_test = [
    {"dmfID": "001", "current_child_count": 1, "keyword":None, "event":"e1"},
    {"dmfID": None, "current_child_count": None, "keyword":None, "event":"e2"},
    {"dmfID": None, "current_child_count": None, "keyword":"delete", "event":"e3"},
    {"dmfID": "001", "current_child_count": 0, "keyword":None, "event":"e4"}
]

if __name__ == "__main__":
    controller = SM_Controller()
    for event in event_test:
        controller.push(event)
    # print(controller.sm.state)