import numpy as np
from paddle import fluid
from paddle.fluid import layers

import _init_paths
from src.fluid_utils import fluid_sequence_advance

def train_rnn(item_fc, h_0, output_type):
    shifted_item_fc = fluid_sequence_advance(item_fc, OOV=0)
    drnn = fluid.layers.DynamicRNN()
    with drnn.block():
        last_item_fc = drnn.step_input(shifted_item_fc)
        cur_h_0 = drnn.memory(init=h_0, need_reorder=True)

        # step_input will remove lod info
        last_item_fc = layers.lod_reset(last_item_fc, cur_h_0)

        next_h_0 = layers.dynamic_gru(last_item_fc, size=4, h_0=cur_h_0, param_attr=fluid.ParamAttr(name="item_gru.w_0"), bias_attr=fluid.ParamAttr(name="item_gru.b_0"))
        if output_type == 'c_Q':
            Q = layers.fc(next_h_0, 1, param_attr=fluid.ParamAttr(name="c_Q_fc.w_0"), bias_attr=fluid.ParamAttr(name="c_Q_fc.b_0"))
            drnn.output(Q)
        elif output_type == 'max_Q':
            Q = layers.fc(next_h_0, 1, param_attr=fluid.ParamAttr(name="max_Q_fc.w_0"), bias_attr=fluid.ParamAttr(name="max_Q_fc.b_0"))
            drnn.output(Q)

        # update
        drnn.update_memory(cur_h_0, next_h_0)

    drnn_output = drnn()
    return drnn_output


def print_params(program, scope):
    for param in program.global_block().all_parameters():
        array = np.array(scope.find_var(param.name).get_tensor())
        print (param.name, array.shape, array.flatten()[:4])


def main():
    place = fluid.CPUPlace()
    scope = fluid.Scope()
    program = fluid.Program()
    start_up = fluid.Program()
    with fluid.scope_guard(scope):
        with fluid.program_guard(program, start_up):
            with fluid.unique_name.guard():
                x = layers.data('x', shape=[-1, 4], lod_level=1)
                h_0 = layers.data('h_0', shape=[-1, 4], lod_level=1)
                item_fc = layers.fc(x, 4*3, param_attr=fluid.ParamAttr(name="item_fc.w_0"), bias_attr=fluid.ParamAttr(name="item_fc.b_0"))

                c_Q = train_rnn(item_fc, h_0, 'c_Q')
                max_Q = train_rnn(item_fc, h_0, 'max_Q')

                opt2 = fluid.optimizer.Adam(learning_rate=0.1)
                opt2.minimize(-1.0 * max_Q, parameter_list=['max_Q_fc.w_0', 'max_Q_fc.b_0', 'item_gru.w_0', 'item_gru.b_0'])

                opt1 = fluid.optimizer.Adam(learning_rate=0.1)
                # opt1.minimize(y1, parameter_list=['y1.w_0', 'y1.b_0'])
                opt1.minimize(c_Q)

        exe = fluid.Executor(place)
        exe.run(start_up)

    print_params(program, scope)


if __name__ == '__main__':
    main()