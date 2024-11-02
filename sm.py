from transitions import Machine, State
from utils import Queue

class Model:
    pass

class StateMachine:
    states = [
        {"name": "idle"},
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

    _count = -1
    _keyword = ""

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

    def __str__(self) -> str:
        return f"State(count:{self.hypothesis_count}, keyword:{self._keyword if self._keyword else 'None'})"

    @property
    def hypothesis_count(self):
        return self._count
    
    @hypothesis_count.setter
    def hypothesis_count(self, value):
        self._count = value

    def check_dmf(self, current_count):
        assert self.state == "dmf_post"
        
        try:
            if (hypothesis_succeed := (self.hypothesis_count == current_count)):
                return self._keyword, self._q
            else:
                return False, None
        finally:
            self.trigger("reset")
    
    def proccess_keyword(self, keyword):
        self._keyword = keyword
        if keyword == "delete":
            self.hypothesis_count -= 1
    
    @property
    def state(self):
        return self._m.state

class SM_Controller():
    def __init__(self):
        self.sm = StateMachine()

    def push(self, event):
        print(f"current_state{str(self.sm)}, pushing event:{event}")

        if event["dmfID"] is not None:
            self.sm.trigger("reach_dmf")
            self.sm.push(event)
            if self.sm.state == "dmf_pre":
                self.sm.hypothesis_count = event["current_child_count"]
            elif self.sm.state == "dmf_post":
                dmf_type, event_trace = self.sm.check_dmf(event["current_child_count"])
                print(f"found DMF {dmf_type}")
        
        if event["keyword"] is not None:
            self.sm.trigger((keyword := event["keyword"]))
            self.sm.push(event)
            self.sm.proccess_keyword(keyword)
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