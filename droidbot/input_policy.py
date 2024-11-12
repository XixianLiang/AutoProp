import sys
import json
import logging
import os
import random
import itertools
import time
import networkx as nx
from abc import abstractmethod

# from droidbot.input_manager import InputManager

from .input_event import InputEvent, KeyEvent, IntentEvent, TouchEvent, ManualEvent, SetTextEvent, KillAppEvent, EnterEvent
from .intent import Intent
from .utg import UTG

from .device_state import DeviceState
from .app import App
from typing import TYPE_CHECKING, Dict, Tuple, List
if TYPE_CHECKING:
    from .input_manager import InputManager
    from .input_event import InputEvent
    from .device import Device
    from .app import App
    from utils import DMF_extractor


from utils import Queue, DMF, DMF_dict, case_insensitive_check, ADD, DELETE, SEARCH, generate_xml_node
from .read_events import get_filenames

# Max number of restarts
MAX_NUM_RESTARTS = 5
# Max number of steps outside the app
MAX_NUM_STEPS_OUTSIDE = 5
MAX_NUM_STEPS_OUTSIDE_KILL = 10
# Max number of replay tries
MAX_REPLY_TRIES = 5

# Some input event flags
EVENT_FLAG_STARTED = "+started"
EVENT_FLAG_START_APP = "+start_app"
EVENT_FLAG_STOP_APP = "+stop_app"
EVENT_FLAG_EXPLORE = "+explore"
EVENT_FLAG_NAVIGATE = "+navigate"
EVENT_FLAG_TOUCH = "+touch"

# Policy taxanomy
POLICY_NAIVE_DFS = "dfs_naive"
POLICY_GREEDY_DFS = "dfs_greedy"
POLICY_NAIVE_BFS = "bfs_naive"
POLICY_GREEDY_BFS = "bfs_greedy"
POLICY_REPLAY = "replay"
POLICY_MANUAL = "manual"
POLICY_MONKEY = "monkey"
POLICY_NONE = "none"
POLICY_MEMORY_GUIDED = "memory_guided"  # implemented in input_policy2
POLICY_LLM_GUIDED = "llm_guided"  # implemented in input_policy3

DMF_PATH = "output_dmf.json"
CACHE_PATH = "cached_events.txt"

Type_output_DMFs = Dict[str, DMF_dict]

class InputInterruptedException(Exception):
    pass



class InputPolicy(object):
    """
    This class is responsible for generating events to stimulate more app behaviour
    It should call AppEventManager.send_event method continuously
    """

    current_state:DeviceState

    def __init__(self, device:"Device", app:"App"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.app = app
        self.action_count = 0
        self.master = None

    def clear_app_data(self):
        r = self.device.adb.shell(f"pm clear {self.app.package_name}")
        assert r == "Success"

    def start(self, input_manager: "InputManager"):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        self.action_count = 0
        self.dmf_dict:Type_output_DMFs = dict()

        from utils import Queue
        self.cache = Queue(max_size=50)
        self.logger.info(f"Cache size is {self.cache._max_size}")

        self.current_state = self.device.get_current_state()
        
        while input_manager.enabled and self.action_count < input_manager.event_count:
            try:
                # * make sure the first event is go to HOME screen
                # * the second event is to start the app
                # if self.action_count == 0 and self.master is None:
                #     event = KeyEvent(name="HOME")
                # elif self.action_count == 1 and self.master is None:
                #     event = IntentEvent(self.app.get_start_intent())
                if self.action_count == 0:
                    self.clear_app_data()
                if self.action_count % 200 == 0 and self.master is None:
                    event = KillAppEvent(app=self.app)
                else:
                    event = self.generate_event()
                if isinstance(event, IntentEvent):
                    pass
                input_manager.add_event(event)
                if self.action_count > 2:
                    self._output_DMF()
                
            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.warning("stop sending events: %s" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()
                continue
            
            
            # TODO caching
            try:
                current_child_count = self.current_state.recydumper.num_children
                current_state_str = self.current_state.state_str
                dmfID = self.current_state.state_str_without_recyclerview[:6]

                self.cache.append({
                    "id": self.action_count, 
                    "state_str": current_state_str,
                    "screenshot_path": self.current_state.screenshot_path,
                    "dmfID":dmfID if current_child_count else None,
                    "current_child_count": current_child_count if current_child_count else None,
                    "text_lt" : self.current_state.recydumper.recyclerView_text if current_child_count else None,
                    "keyword": event.keyword,
                    "event_view": json.dumps(event.view),
                    "action": event.event_type,
                    "text": event.text if event.event_type == "set_text" else "None"
                    # "view_list": str(self.current_state.views_without_recyclerview)
                })
                # a = self.cache.get_dmf_dict()
                if event.event_type == "set_text":
                    pass
                with open(CACHE_PATH, "w") as fp:
                    # fp.writelines(f"{line}\n" for line in self.cache)
                    fp.writelines(f"{json.dumps(line)}\n" for line in self.cache)
                
                self._output_DMF()

            except AttributeError as e:
                self.logger.warning("exeception during caching events")
                import traceback
                traceback.print_exc()
            self.action_count += 1
    
    def _output_DMF(self):
        """
        save the found DMF into a json file
        """
        with open(CACHE_PATH, "r") as fp:
            q = Queue([json.loads(_) for _ in fp])
            res:dict[str, "DMF_extractor"] = q.get_dmf_dict()

            for _dmfID, _dmf in res.items():
                self.dmf_dict[_dmfID] = self.dmf_dict.get(_dmfID, DMF_dict())
                self.dmf_dict[_dmfID][_dmf.keyword] = self.dmf_dict[_dmfID].get(_dmf.keyword, _dmf)

        with open(DMF_PATH, "w") as fp:
            json.dump(self.dmf_dict, fp)   
        
        self.__output_dot_fmt_DMF()
       
        
        pass
    
    def _output_dot_fmt_DMF(self, shortcuts=[], filename="output_dmf.txt"):
        """
        save the found DMF to a txt
        """
        lines = []
        with open(filename, "w") as fp:
            if self.dmf_dict:
                for dmfID, DMFs in self.dmf_dict.items():
                    for keyword, _dmf in DMFs.items():
                        _dmf["keyword"] = keyword
                        _dmf = DMF(_dmf)
                        # DMF信息
                        lines.append("%s::%s::%s::\n" % (keyword, dmfID, _dmf.changed_item))
                        # 三行占位
                        lines.extend(["====\n" for _ in range(3)])
                        # 事件信息
                        # for i, _ in enumerate(dmf_trace["event_trace"]):
                        #     event_view = json.loads(_["event_view"])
                        #     event_text = str(event_view["text"])
                        #     action = _["action"]
                        #     action = "click" if action == "touch" else action
                        #     action = "edit" if action == "set_text" else action
                        #     lines.append("%s::%s::%s::%s::\n" % (i+1, action, event_text, generate_xml_node(event_view)))
                        j = 1
                        for i, _ in enumerate(self.cache):
                            # 如果在shortcut里，不考虑
                            if any(i in range(shortcut[0], shortcut[1]) for shortcut in shortcuts):
                                continue
                            event_view = json.loads(_["event_view"])
                            event_text = _["text"]
                            action = _["action"]
                            action = "click" if action == "touch" else action
                            action = "edit" if action == "set_text" else action
                            lines.append("%s::%s::%s::%s::\n" % (j, action, event_text, generate_xml_node(event_view)))
                            j += 1
                fp.writelines(lines)

    @abstractmethod
    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass

class FuzzingPolicy(InputPolicy):
    """
    DFS/BFS (according to search_method) strategy to explore UFG (new)
    """

    def __init__(self, device, app):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.app = app
        self.action_count = 0
        self.master = None

    def start(self, input_manager):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        self.action_count = 0
        
        while input_manager.enabled and self.action_count < input_manager.event_count:
            try:
                # # make sure the first event is go to HOME screen
                # # the second event is to start the app
                # if self.action_count == 0 and self.master is None:
                #     event = KeyEvent(name="HOME")
                # elif self.action_count == 1 and self.master is None:
                #     event = IntentEvent(self.app.get_start_intent())
                if self.action_count == 0 and self.master is None:
                    event = KillAppEvent(app=self.app)
                else:
                    event = self.generate_event()
                input_manager.add_event(event)
            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.warning("stop sending events: %s" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()
                continue
            self.action_count += 1
    

    @abstractmethod
    def generate_event(self):
        """
        generate an event
        @return:
        """
        current_state = self.device.get_current_state()
        print("Current state: %s" % current_state.state_str)

class ReplayPolicy(InputPolicy):
    def __init__(self, device, app, events_path):
        super().__init__(device, app)
        self.events_path = events_path
    
    def read_events(self):
        filenames = get_filenames(self.events_path)
        self.input_events = []
        for i, filename in enumerate(filenames):
            with open(filepath := os.path.join(self.events_path, filename), "r") as fp:
                try:
                    self.input_events.append(json.load(fp))
                except:
                    self.logger.error(f"error when decoding file: {filepath}")

    def get_dmf_info(self) -> Tuple[str, list[dict], Dict]:
        """
        :return: dmf_start_states, dmf_events, dmf_dict
        """
        with open(DMF_PATH, "r") as fp:
            dmf_dict:Type_output_DMFs = json.load(fp)
        
        # 获取所有dmf的起始页面，在replay时确认能回到这个页面
        dmf_start_states = set()

        for _dmfID, _dmfs in dmf_dict.items():
            for _dmf_type, dmf in _dmfs.items():
                # dmf_start_states.add(dmf["event_trace"][0]["state_str"])
                dmf_start_states.add(dmf["event_trace"][0]["state_str"])
                dmf_events = dmf["event_trace"]
                
        
        return dmf_start_states, dmf_events, dmf_dict
    


    def start(self, input_manager: "InputManager"):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        self.action_count = 0
        
        self.read_events()
        self.dmf_start_states, self.dmf_events, self.dmf_dict = self.get_dmf_info()

        last_event_of_dmf = self.dmf_events[-1]
        target_end_index = last_event_of_dmf["id"]

        # # replay 成功debug的时候先跳过，这是用来验证序列有没有问题的
        # self.logger.warning("trying to replay all events")
        # # print(self.replay(input_manager, end_index=target_end_index))
        
        # self.logger.warning("trying to reduce")
        # shortcuts = self.reduce(input_manager, end_index=target_end_index)

        # self.logger.warning("trying to replay the reduced event")
        # reduce_successed = self.replay(input_manager, (pair:=shortcuts.pop()), end_index=target_end_index)

        # self.logger.warning(f"{'reduce_successed' if reduce_successed else 'reduce_failed'}")
        # self.logger.warning(f"found shortcut {pair}")


        self._output_dot_fmt_DMF(shortcuts=[(7, 12)], filename="reduced_trace.txt")
    
    def _output_dot_fmt_DMF(self, shortcuts=[], filename="output_dmf.txt"):
        """
        save the found DMF to a txt
        """
        lines = []
        with open(filename, "w") as fp:
            if self.dmf_dict:
                for dmfID, DMFs in self.dmf_dict.items():
                    for keyword, _dmf in DMFs.items():
                        _dmf["keyword"] = keyword
                        _dmf = DMF(_dmf)
                        # DMF信息
                        lines.append("%s::%s::%s::\n" % (keyword, dmfID, _dmf.changed_item))
                        # 三行占位
                        lines.extend(["====\n" for _ in range(3)])
                        # 事件信息
                        # for i, _ in enumerate(dmf_trace["event_trace"]):
                        #     event_view = json.loads(_["event_view"])
                        #     event_text = str(event_view["text"])
                        #     action = _["action"]
                        #     action = "click" if action == "touch" else action
                        #     action = "edit" if action == "set_text" else action
                        #     lines.append("%s::%s::%s::%s::\n" % (i+1, action, event_text, generate_xml_node(event_view)))
                        j = 1
                        for i, _ in enumerate(self.input_events):
                            # 如果在shortcut里，不考虑
                            if any(i in range(shortcut[0], shortcut[1]) for shortcut in shortcuts):
                                continue
                            event = _["event"]
                            action = event["event_type"]

                            if action in ["kill_app", "intent"]:
                                continue

                            action = "click" if action == "touch" else action
                            action = "edit" if action == "set_text" else action
                            event_view = event["view"]
                            event_text = event_view["text"]
                            lines.append("%s::%s::%s::%s::\n" % (j, action, event_text, generate_xml_node(event_view)))
                            j += 1
                fp.writelines(lines)

    def reduce(self, input_manager:"InputManager", end_index:int) -> list[Tuple[int,int]]:
        # 获取dmf对应事件的id，编排组合并排序，适应算法，最后获得的是削减的candidates list。
        # 示例 event_ids = [5, 6, 7, 8]      则 candidates 为 [(5, 8), (6, 8), (7, 8), (5, 7), (6, 7), (5, 6)]
        # 最后获得的candidates是可行的
        event_ids = [_["id"] for _ in self.dmf_events[:-1]]

        # event_range = (event_ids[0], event_ids[-1])
        candidates = list(itertools.combinations(event_ids, 2))
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        candidates
        shortcuts = []

        for pair in candidates:
            replay_result = self.replay(input_manager, pair, end_index)
            if replay_result:
                shortcuts.append(pair)
                self.clean_candidates(pair, candidates)
        
        if shortcuts:
            self.logger.warning(f"Reduce success. Available short cuts {shortcuts}")
        # 此时是所有可行的shorcuts
        return shortcuts
        


        
    def clean_candidates(self, pair, candidates:list[Tuple[int, int]]):
        removable = []
        for _ in candidates:
            if pair[0] <= _[0] and not pair[1] <= _[0]:
                removable.append(_)
            elif pair[1] >= _[1] and not pair[0] >= _[1]:
                removable.append(_)
            elif pair[0] >= _[0] and pair[1] <= _[1]:
                removable.append(_)
        
        for _ in set(removable):
            candidates.remove(_)

    def event_in_view(self, event:"InputEvent", event_list:List["InputEvent"]):
        for e in event_list:
            if event.event_type == e.event_type and event.view["signature"] == e.view["signature"]:
                return True
            if event.event_type == e.event_type and event.event_type == "set_text":
                return event.view["content_free_signature"] == e.view["content_free_signature"]
        return False
    
    def replay(self, input_manager:"InputManager", pair:Tuple[int, int]=None, end_index=None):
        self.clear_app_data()
        time.sleep(3)
        current_state = self.device.get_current_state()
        ## 根据short cut重放
        if pair and end_index:
            self.logger.info(f"replaying with shortcut {pair}")
            for i, event_record in enumerate(self.input_events):
                try:
                    self.logger.info(f"choose event {i}, shortcut {pair}")
                    # 回放到dmf的最后一个事件，则结束
                    if i >= pair[0] and i < pair[1] :
                        continue

                    if i == end_index:
                        break

                    event = InputEvent.from_dict(event_record["event"])

                    if current_state.foreground_activity is None:
                        pass
                    # if pair == (7, 12):
                    #     pass
                    if current_state.foreground_activity.startswith(self.app.package_name) and \
                        not self.event_in_view(event, current_state.get_possible_input()):
                        return False
                        
                    input_manager.add_event(event)
                    current_state = self.device.get_current_state()
                    replay_success = current_state.state_str == event_record["stop_state"]

                    self.logger.info(f"replayed event {i}, replayed {'succeess' if replay_success else 'fail'}")

                    if not (replay_success := current_state.state_str == event_record["stop_state"]):
                        self.logger.warning("Path Deviated, kept executing")
                        # input_manager.add_event(KillAppEvent(app=self.app))
                        # input_manager.add_event(IntentEvent(self.app.get_start_intent()))
                    
                except KeyboardInterrupt:
                    break
                except InputInterruptedException as e:
                    self.logger.warning("stop sending events: %s" % e)
                    break
                except Exception as e:
                    self.logger.warning("exception during sending events: %s" % e)
                    import traceback
                    traceback.print_exc()
                    continue
                
                self.action_count += 1
            
            return replay_success


        ## 全量replay
        elif end_index:
            for i, event_record in enumerate(self.input_events):
                try:
                    self.logger.info(f"replaying event {i}, end index {end_index}")
                    # 回放到dmf的最后一个事件，则结束
                    if i == end_index:
                        break

                    event = InputEvent.from_dict(event_record["event"])
                    # # 如果这个event没有改变页面，跳过之。
                    # if event_record["start_state"] == event_record["stop_state"]:
                    #     continue
                    input_manager.add_event(event)

                    current_state = self.device.get_current_state()
                    if not (replay_success := current_state.state_str == event_record["stop_state"]):
                        self.logger.warning("Path Deviated, kept executing")
                    
                except KeyboardInterrupt:
                    break
                except InputInterruptedException as e:
                    self.logger.warning("stop sending events: %s" % e)
                    break
                except Exception as e:
                    self.logger.warning("exception during sending events: %s" % e)
                    import traceback
                    traceback.print_exc()
                    continue
                
                self.action_count += 1
            
            return replay_success
        
        else:
            raise AttributeError("You have to assign pair or end_index arg")
        
        



class NoneInputPolicy(InputPolicy):
    """
    do not send any event
    """

    def __init__(self, device, app):
        super(NoneInputPolicy, self).__init__(device, app)

    def generate_event(self):
        """
        generate an event
        @return:
        """
        return None


class UtgBasedInputPolicy(InputPolicy):
    """
    state-based input policy
    """

    def __init__(self, device, app, random_input):
        super(UtgBasedInputPolicy, self).__init__(device, app)
        self.random_input = random_input
        self.script = None
        self.master = None
        self.script_events = []
        self.last_event = None
        self.last_state = None
        self.current_state = None
        self.utg = UTG(device=device, app=app, random_input=random_input)
        self.script_event_idx = 0
        if self.device.humanoid is not None:
            self.humanoid_view_trees = []
            self.humanoid_events = []

    def generate_event(self):
        """
        generate an event
        @return:
        """

        # Get current device state
        self.current_state = self.device.get_current_state()
        if self.current_state is None:
            import time
            time.sleep(3)
            self.logger.warning("Current state is None, trying BACK to navigate to normal state.")
            return KeyEvent(name="BACK")

        self.utg.add_transition(self.last_event, self.last_state, self.current_state)

        event = self.generate_event_based_on_utg()

        self.last_state = self.current_state
        self.last_event = event
        return event


    @abstractmethod
    def generate_event_based_on_utg(self):
        """
        generate an event based on UTG
        :return: InputEvent
        """
        pass


class UtgNaiveSearchPolicy(UtgBasedInputPolicy):
    """
    depth-first strategy to explore UFG (old)
    """

    def __init__(self, device, app, random_input, search_method):
        super(UtgNaiveSearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.explored_views = set()
        self.state_transitions = set()
        self.search_method = search_method

        self.last_event_flag = ""
        self.last_event_str = None
        self.last_state = None

        self.search_state = None
        self.add_state = None

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next", "delete", "remove", "add", "search"]

    def generate_event_based_on_utg(self):
        """
        generate an event based on current device state
        note: ensure these fields are properly maintained in each transaction:
          last_event_flag, last_touched_view, last_state, exploited_views, state_transitions
        @return: InputEvent
        """
        self.save_state_transition(self.last_event_str, self.last_state, self.current_state)

        if self.device.is_foreground(self.app):
            # the app is in foreground, clear last_event_flag
            self.last_event_flag = EVENT_FLAG_STARTED
        else:
            number_of_starts = self.last_event_flag.count(EVENT_FLAG_START_APP)
            # If we have tried too many times but the app is still not started, stop DroidBot
            if number_of_starts > MAX_NUM_RESTARTS:
                raise InputInterruptedException("The app cannot be started.")

            # if app is not started, try start it
            if self.last_event_flag.endswith(EVENT_FLAG_START_APP):
                # It seems the app stuck at some state, and cannot be started
                # just pass to let viewclient deal with this case
                self.logger.info("The app had been restarted %d times.", number_of_starts)
                self.logger.info("Trying to restart app...")
                pass
            else:
                start_app_intent = self.app.get_start_intent()

                self.last_event_flag += EVENT_FLAG_START_APP
                self.last_event_str = EVENT_FLAG_START_APP
                return IntentEvent(start_app_intent)

        # select a view to click
        view_to_touch = self.select_a_view(self.current_state)

        # if no view can be selected, restart the app
        if view_to_touch is None:
            stop_app_intent = self.app.get_stop_intent()
            self.last_event_flag += EVENT_FLAG_STOP_APP
            self.last_event_str = EVENT_FLAG_STOP_APP
            return IntentEvent(stop_app_intent)

        view_to_touch_str = view_to_touch['view_str']
        if view_to_touch_str.startswith('BACK'):
            result = KeyEvent('BACK')
        else:
            result = TouchEvent(view=view_to_touch)

        self.last_event_flag += EVENT_FLAG_TOUCH
        self.last_event_str = view_to_touch_str
        self.save_explored_view(self.current_state, self.last_event_str)
        return result

    def select_a_view(self, state):
        """
        select a view in the view list of given state, let droidbot touch it
        @param state: DeviceState
        @return:
        """
        views = []
        for view in state.views:
            if view['enabled'] and len(view['children']) == 0:
                views.append(view)

        if self.random_input:
            random.shuffle(views)

        # add a "BACK" view, consider go back first/last according to search policy
        mock_view_back = {'view_str': 'BACK_%s' % state.foreground_activity,
                          'text': 'BACK_%s' % state.foreground_activity}
        if self.search_method == POLICY_NAIVE_DFS:
            views.append(mock_view_back)
        elif self.search_method == POLICY_NAIVE_BFS:
            views.insert(0, mock_view_back)

        # first try to find a preferable view
        for view in views:
            view_text = view['text'] if view['text'] is not None else ''
            view_text = view_text.lower().strip()
            if view_text in self.preferred_buttons \
                    and (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an preferred view: %s" % view['view_str'])
                return view

        # try to find a un-clicked view
        for view in views:
            if (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an un-clicked view: %s" % view['view_str'])
                return view

        # if all enabled views have been clicked, try jump to another activity by clicking one of state transitions
        if self.random_input:
            random.shuffle(views)
        transition_views = {transition[0] for transition in self.state_transitions}
        for view in views:
            if view['view_str'] in transition_views:
                self.logger.info("selected a transition view: %s" % view['view_str'])
                return view

        # no window transition found, just return a random view
        # view = views[0]
        # self.logger.info("selected a random view: %s" % view['view_str'])
        # return view

        # DroidBot stuck on current state, return None
        self.logger.info("no view could be selected in state: %s" % state.tag)
        return None

    def save_state_transition(self, event_str, old_state, new_state):
        """
        save the state transition
        @param event_str: str, representing the event cause the transition
        @param old_state: DeviceState
        @param new_state: DeviceState
        @return:
        """
        if event_str is None or old_state is None or new_state is None:
            return
        if new_state.is_different_from(old_state):
            self.state_transitions.add((event_str, old_state.tag, new_state.tag))

    def save_explored_view(self, state, view_str):
        """
        save the explored view
        @param state: DeviceState, where the view located
        @param view_str: str, representing a view
        @return:
        """
        if not state:
            return
        state_activity = state.foreground_activity
        self.explored_views.add((state_activity, view_str))


class UtgGreedySearchPolicy(UtgBasedInputPolicy):
    """
    DFS/BFS (according to search_method) strategy to explore UFG (new)
    """

    def __init__(self, device, app, random_input, search_method):
        super(UtgGreedySearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_method = search_method

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

        self.__nav_target = None
        self.__nav_num_steps = -1
        self.__num_restarts = 0
        self.__num_steps_outside = 0
        self.__event_trace = ""
        self.__missed_states = set()

        self.search_state = None
        self.add_state = None

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.current_state
        self.logger.info("Current state: %s" % current_state.state_str)
        if current_state.state_str in self.__missed_states:
            self.__missed_states.remove(current_state.state_str)

        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            # It seems the app stucks at some state, has been
            # 1) force stopped (START, STOP)
            #    just start the app again by increasing self.__num_restarts
            # 2) started at least once and cannot be started (START)
            #    pass to let viewclient deal with this case
            # 3) nothing
            #    a normal start. clear self.__num_restarts.

            if self.__event_trace.endswith(EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP) \
                    or self.__event_trace.endswith(EVENT_FLAG_START_APP):
                self.__num_restarts += 1
                self.logger.info("The app had been restarted %d times.", self.__num_restarts)
            else:
                self.__num_restarts = 0

            # pass (START) through
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP):
                if self.__num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                else:
                    # Start the app
                    self.__event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self.__num_steps_outside += 1

            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            # If the app is in foreground
            self.__num_steps_outside = 0

        # Get all possible input events
        possible_events = current_state.get_possible_input()
        # possible_events = current_state.get_all_input()

        random.shuffle(possible_events)

        # * DMF policy
        # if there's a edit of click event, do it first.
        # print([e.event_type for e in possible_events])
        # possible_events = sorted(possible_events, key=lambda event: event.event_type in ["touch", "set_text", "long_touch"], reverse=True)
        # print([e.event_type for e in possible_events])
        # random.shuffle(possible_events)


        
        #* Search
        # if any("search" in (target_event := event).get_signature() for event in possible_events) and random.random() > 0.15:
        if not self.search_state and any("search" in (target_event := event).get_signature().lower() for event in possible_events) and random.random() > 0.8:
            if len(text_lt := self.current_state.recydumper.recyclerView_text) < 2 or self.search_state is not None:
                pass
            else:
                target_event.keyword = SEARCH + "_" + random.choice(text_lt)
                self.search_state:str = target_event.keyword
                return target_event
        
        if self.search_state and self.search_state != "Enter_search":
            input_text = self.search_state.split("_", 1)[-1]
            self.search_state = "Enter_search"
            return self.current_state.get_search_settextEvent(input_text)
        
        if self.search_state == "Enter_search":
            self.search_state = None
            return EnterEvent()
        
        #* Delete
        DELETE_KEYWORD = ["Trash", "Delete"]
        delete_events = []
        for KEYWORD in DELETE_KEYWORD:
            if target_event := self.get_event(KEYWORD, possible_events, propability=0.6):
                target_event.keyword = DELETE
                delete_events.append(target_event)

        #* add
        add_events = []
        if not self.add_state:
            if (target_event := self.get_event("fab", possible_events, propability=0.8)):
                target_event.keyword = ADD
                self.add_state = "did_fab"
                return target_event
        else:
            if (target_event := self.get_event("fab", possible_events, propability=0.5)):
                target_event.keyword = ADD
                add_events.append(target_event)

        # if (target_event := self.get_event("add", possible_events, propability=0.8)):
        #     target_event.keyword = ADD
        #     add_events.append(target_event)

        if self.add_state == "did_fab":
            # reduce 测试
            if (target_event := self.get_event(SetTextEvent, possible_events, propability=0.8)):
                self.add_state = "set_text" if random.random() > 0.5 else self.add_state
                return target_event
        
        if self.add_state == "set_text":
            if (target_event := self.get_event("save", possible_events, propability=0.8)):
                self.add_state = None
                return target_event
            add_events.append(KeyEvent(name="BACK"))
        
        ADD_EVENTS = [SetTextEvent,  "save", "ok"]
        for e in ADD_EVENTS:
            if (target_event := self.get_event(e, possible_events, propability=0.7)):
                add_events.append(target_event)


        
        if random.random() > 0.3:
            if len(add_events) > 0:
                return random.choice(add_events)
            if len(delete_events) > 0:
                return random.choice(delete_events)
        

        
        self.logger.info(f"current text list: {self.current_state.recydumper.recyclerView_text}")

        if self.search_method == POLICY_GREEDY_DFS:
            possible_events.append(KeyEvent(name="BACK"))
        elif self.search_method == POLICY_GREEDY_BFS:
            possible_events.insert(0, KeyEvent(name="BACK"))

        # If there is an unexplored event, try the event first
        all_explored = True
        for input_event in possible_events:
            if not self.utg.is_event_explored(event=input_event, state=current_state):
                all_explored = False
                self.logger.info("Trying an unexplored event.")
                self.__event_trace += EVENT_FLAG_EXPLORE
                return input_event

        # target_state = self.__get_nav_target(current_state)
        # if target_state:
        #     navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=target_state)
        #     if navigation_steps and len(navigation_steps) > 0:
        #         self.logger.info("Navigating to %s, %d steps left." % (target_state.state_str, len(navigation_steps)))
        #         self.__event_trace += EVENT_FLAG_NAVIGATE
        #         return navigation_steps[0][1]

        # if self.__random_explore:
        #     self.logger.info("Trying random event.")
        #     random.shuffle(possible_events)
        #     return possible_events[0]

        # if All events explored try a random event
        if all_explored:
            self.logger.info("Cannot find an exploration target. Trying a random event...")
            random.shuffle(possible_events)
            return possible_events[0]

        # # If couldn't find a exploration target, stop the app
        # stop_app_intent = self.app.get_stop_intent()
        # self.logger.info("Cannot find an exploration target. Trying to restart app...")
        # self.__event_trace += EVENT_FLAG_STOP_APP
        # return IntentEvent(intent=stop_app_intent)

    def get_event(self, keyword, possible_events:list[InputEvent], propability=0.35):
        if random.random() > propability:
            return None
        
        if isinstance(keyword, str):
            if any(keyword.lower() in (target_event := event).get_signature().lower() for event in possible_events):
                return target_event
        elif issubclass(keyword, InputEvent):
            if any(isinstance(target_event := event, keyword) for event in possible_events):
                return target_event
        else:
            raise AttributeError(f"keyword should be str or class/subclass of InputEvent, your keyword is {keyword}")


    def __get_nav_target(self, current_state):
        # If last event is a navigation event
        if self.__nav_target and self.__event_trace.endswith(EVENT_FLAG_NAVIGATE):
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if navigation_steps and 0 < len(navigation_steps) <= self.__nav_num_steps:
                # If last navigation was successful, use current nav target
                self.__nav_num_steps = len(navigation_steps)
                return self.__nav_target
            else:
                # If last navigation was failed, add nav target to missing states
                self.__missed_states.add(self.__nav_target.state_str)

        reachable_states = self.utg.get_reachable_states(current_state)
        if self.random_input:
            random.shuffle(reachable_states)

        for state in reachable_states:
            # Only consider foreground states
            if state.get_app_activity_depth(self.app) != 0:
                continue
            # Do not consider missed states
            if state.state_str in self.__missed_states:
                continue
            # Do not consider explored states
            if self.utg.is_state_explored(state):
                continue
            self.__nav_target = state
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if len(navigation_steps) > 0:
                self.__nav_num_steps = len(navigation_steps)
                return state

        self.__nav_target = None
        self.__nav_num_steps = -1
        return None

class UtgReplayPolicy(InputPolicy):
    """
    Replay DroidBot output generated by UTG policy
    """

    def __init__(self, device, app, replay_output):
        super(UtgReplayPolicy, self).__init__(device, app)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.replay_output = replay_output

        import os
        event_dir = os.path.join(replay_output, "events")
        self.event_paths = sorted([os.path.join(event_dir, x) for x in
                                   next(os.walk(event_dir))[2]
                                   if x.endswith(".json")])
        # skip HOME and start app intent
        self.device = device
        self.app = app
        self.event_idx = 2
        self.num_replay_tries = 0
        self.utg = UTG(device=device, app=app, random_input=None)
        self.last_event = None
        self.last_state = None
        self.current_state = None

    def generate_event(self):
        """
        generate an event based on replay_output
        @return: InputEvent
        """
        import time
        while self.event_idx < len(self.event_paths) and \
              self.num_replay_tries < MAX_REPLY_TRIES:
            self.num_replay_tries += 1
            current_state = self.device.get_current_state()
            if current_state is None:
                time.sleep(5)
                self.num_replay_tries = 0
                return KeyEvent(name="BACK")

            curr_event_idx = self.event_idx
            self.__update_utg()
            while curr_event_idx < len(self.event_paths):
                event_path = self.event_paths[curr_event_idx]
                with open(event_path, "r") as f:
                    curr_event_idx += 1

                    try:
                        event_dict = json.load(f)
                    except Exception as e:
                        self.logger.info("Loading %s failed" % event_path)
                        continue

                    if event_dict["start_state"] != current_state.state_str:
                        continue
                    if not self.device.is_foreground(self.app):
                        # if current app is in background, bring it to foreground
                        component = self.app.get_package_name()
                        if self.app.get_main_activity():
                            component += "/%s" % self.app.get_main_activity()
                        return IntentEvent(Intent(suffix=component))
                    
                    self.logger.info("Replaying %s" % event_path)
                    self.event_idx = curr_event_idx
                    self.num_replay_tries = 0
                    # return InputEvent.from_dict(event_dict["event"])
                    event = InputEvent.from_dict(event_dict["event"])
                    self.last_state = self.current_state
                    self.last_event = event
                    return event                    

            time.sleep(5)

        # raise InputInterruptedException("No more record can be replayed.")
    def __update_utg(self):
        self.utg.add_transition(self.last_event, self.last_state, self.current_state)



class ManualPolicy(UtgBasedInputPolicy):
    """
    manually explore UFG
    """

    def __init__(self, device, app):
        super(ManualPolicy, self).__init__(device, app, False)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.__first_event = True

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        if self.__first_event:
            self.__first_event = False
            self.logger.info("Trying to start the app...")
            start_app_intent = self.app.get_start_intent()
            return IntentEvent(intent=start_app_intent)
        else:
            return ManualEvent()
