

class IMIRDB:
    def get_by_mir_state(self, mir_state):
        raise Exception("Must be implemented")

    def get_by_tag(self, tag):
        raise Exception("Must be implemented")

    def get_by_string(self, query):
        raise Exception("Must be implemented")
#()