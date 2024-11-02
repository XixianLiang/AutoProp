import copy

RECYCLERVIEW_ID = "androidx.recyclerview.widget.RecyclerView"

class RecyclerViewDumper:
    def __init__(self, view_tree, views) -> None:
        """
        param:
        view_tree: raw view tree
        views: raw views list
        """
        self._view_tree_without_recyclerview = copy.deepcopy(view_tree)
        self._views_without_recyclerview:list[dict] = copy.deepcopy(views)
        self._recyclerView_texts = list()

        self.num_children = None
        self.root_id = None

        self.proccess_tree()

    def proccess_tree(self):
        def remove_widget(temp_id):
            """
            remove widget according to the temp_id
            """
            for item in self._views_without_recyclerview:
                if item["temp_id"] == temp_id:
                    self._views_without_recyclerview.remove(item)

        def dfs_clean_recyclerview(node, is_subtree):
            """
            Use a depth first search algorithm to traverse the tree, 
            and prune the recyclerview node and its succesors in the tree.
            """

            if node is None:
                return
            
            # delete the node from the list if current view is a subtree of recyclerview
            # But don't delete the root node of this subtree
            if is_subtree and node["class"] != RECYCLERVIEW_ID:
                temp_id = node["temp_id"]
                remove_widget(temp_id)

            # recycler_view_node = None
            for child in node["children"]:
                if child["class"] == RECYCLERVIEW_ID:
                    self.num_children = child["child_count"]
                    self.root_id = child["temp_id"]
                    # self.current_child_count = child["child_count"]
                    node["child_count"] -= 1
                    # recycler_view_node = child
                    # if node["text"] is not None:
                    
                    # get the child list
                    for item in (recyclerList := child["children"]):
                        rep_text = self.get_first_text(item)
                        self._recyclerView_texts.append(rep_text)

                    dfs_clean_recyclerview(child, is_subtree=True)
                    continue
                dfs_clean_recyclerview(child, is_subtree=is_subtree)
            # if recycler_view_node:
            #     self.node1 = node
            #     node["child_count"] -= 1
            #     node["children"].remove(recycler_view_node)

            
        dfs_clean_recyclerview(self._view_tree_without_recyclerview, is_subtree=False)
    
    def get_first_text(self, node):
        if node["text"] is not None:
            return node["text"]
        
        for child in node["children"]:
            if (rep_text := self.get_first_text(child)) is not None:
                return rep_text

    # def get_first_text(self):

    @property
    def view_tree_without_recyclerview(self):
        return self._view_tree_without_recyclerview
    
    @property
    def views_without_recyclerview(self):
        return self._views_without_recyclerview
    
    @property
    def recyclerView_text(self):
        return self._recyclerView_texts