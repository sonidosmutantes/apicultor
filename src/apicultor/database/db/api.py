

class MirDbApi:
    """
        MIR Database API Access Interface
    """

    def search_by_content(self, api_key="", content=""):
        """
            Returns a json
        """
        raise Exception("This is an interface with no implementation")

    def search_by_content(self, content=""):
        raise Exception("This is an nterface with no implementation")
    
    def download_by_content(self, content=""):
        raise Exception("This is an nterface with no implementation")
    
    def search_by_mir(self, mir_state):
        raise Exception("This is an nterface with no implementation")

    def get_simillar_sound(self, sound):
        raise Exception("This is an nterface with no implementation")
    
    #TODO: add as a parameter and callback function processing in other methods
    def json_to_id_list(self, json_content):
        """
            Input: json
            Output: id list
        """
    # def get_by_mir_state(self, mir_state):
    #     raise Exception("Must be implemented")

    # def get_by_tag(self, tag):
    #     raise Exception("Must be implemented")

    # def get_by_string(self, query):
    #     raise Exception("Must be implemented")
#()