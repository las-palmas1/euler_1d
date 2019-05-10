

class Logger:
    def __init__(self, level='info', fname='', console=True):
        self.level = level
        self.fname = fname
        self.console = console
        if fname:
            f = open(fname, 'wt')
            f.close()

    def _print(self, mes):
        if self.console:
            print(mes)
        if self.fname:
            with open(self.fname, 'at') as f:
                f.write(mes + '\n')

    def info(self, mes):
        if self.level == 'info' or self.level == 'debug':
            self._print(mes)

    def debug(self, mes):
        mes = 'DEBUG:  %s' % mes
        if self.level == 'debug':
            self._print(mes)
