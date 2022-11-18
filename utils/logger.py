_GLOBAL_LOGGER = None


def get_logger(rank, print_ranks):
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = Logger(rank, print_ranks)

    return _GLOBAL_LOGGER


class Logger(object):
    """
    日志类
    """

    def __init__(self, rank, print_ranks):
        self.rank = rank
        self.print_ranks = print_ranks

        self.m = dict()

    def register_metric(self, metric_key, meter, print_format=None, reset_after_print=False):
        pass

    def print_metrics(self, print_ranks=None):
        fields = []
        for m in self.m.values():
            meter = m["meter"]
            print_format = m["print_format"]
            result = meter.get()
            if isinstance(result, (list, tuple)):
                field = print_format.format(*result)
            else:
                field = print_format.format(result)
            fields.append(field)
            if m["reset_after_print"]:
                meter.reset()

        do_print = self.rank in (print_ranks or self.print_ranks)
        if do_print:
            print("[rank:{}] {}".format(self.rank, ", ".join(fields)))

    def print(self, *args, print_ranks=None):
        # do_print = self.rank in (print_ranks or self.print_ranks)  # 前者为None，则判断后者
        # if do_print:
        print(*args)


