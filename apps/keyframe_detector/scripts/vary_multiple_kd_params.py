#!/usr/bin/env python3
"""Varies multiple keyframe detector parameters """

import argparse

import vary_kd_params


def __parse_args(parser, validate_func):
    """Returns an object containing the arguments to this program """
    if parser is None:
        parser = argparse.ArgumentParser(description="Varies multiple keyframe detector parameters")

    parser.add_argument(
        "--vary-sel-secondary",
        action="store_true",
        help="Vary the selectivity as well.")
    parser.add_argument(
        "--vary-buf-len-secondary",
        action="store_true",
        help="Vary the buffer length as well.")
    parser.add_argument(
        "--vary-levels-secondary",
        action="store_true",
        help="Vary the number of levels as well.")

    args = vary_kd_params.parse_args(parser, validate_func=None)
    vary_sel = args.vary_sel_secondary
    vary_buf_len = args.vary_buf_len_secondary
    vary_levels = args.vary_levels_secondary

    vary_kd_params.validate_file(vary_sel, args.sels_file, "--sels-file")
    vary_kd_params.validate_file(vary_buf_len, args.buf_lens_file, "--buf-lens-file")
    vary_kd_params.validate_file(vary_levels, args.levels_file, "--levels-file")

    if not (vary_sel or vary_buf_len or vary_levels):
        raise Exception("Must specify either \"--vary-sel-secondary\", "
                        "\"--vary-buf-len-secondary\", or \"--vary-levels-secondary\"!")

    if validate_func is not None:
        validate_func(args)

    return args


def run(args):
    """Runs the keyframe detector, varying two parameters according to args """
    vary_sel = args.vary_sel_secondary
    vary_buf_len = args.vary_buf_len_secondary
    vary_levels = args.vary_levels_secondary

    values = []
    if vary_sel:
        values = args.sels
    elif vary_buf_len:
        values = args.buf_lens
    elif vary_levels:
        values = args.nums_levels
    else:
        raise Exception("Must vary something!")

    for value in values:
        if vary_sel:
            args.sel = value
        elif vary_buf_len:
            args.buf_len = value
        elif vary_levels:
            args.levels = value
        else:
            raise Exception("Must vary something!")

        suffix = vary_kd_params.get_key(vary_sel, vary_buf_len, vary_levels) + "_" + str(value)
        vary_kd_params.run(args, suffix)

def __main():
    run(__parse_args(parser=None, validate_func=None))


if __name__ == "__main__":
    __main()
