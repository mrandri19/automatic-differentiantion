class PPrinter:
    def __init__(self):
        self.global_indent = ""
        self.global_indent_level = 0

    def indent(self):
        self.global_indent_level += 4
        self.global_indent = self.global_indent_level * " "
        return ""

    def outdent(self):
        self.global_indent_level -= 4
        self.global_indent = self.global_indent_level * " "
        return ""

    def nl(self):
        return "\n"
