import oneflow as flow
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
def watch_handler(y: flow.typing.Numpy):
    sub_img = y[0,0,:,:]
    sub_img = 1.0 / (1+np.exp(-1 * sub_img))
    sub_img = np.round(sub_img*255)
    # cv2.imwrite('sub_image.jpg', sub_img)
    print("out", np.sum(y))

def watch_step(y: flow.typing.Numpy):
    print("step", np.sum(y))

def watch_weight(y: flow.typing.Numpy):
    print("weight", np.sum(y))

class DLABuilder(object):
    def __init__(self, levels, channels, weight_regularizer, trainable=True, training=True, channel_last=False):
        self.data_format = "NHWC" if channel_last else "NCHW"
        self.weight_initializer = flow.variance_scaling_initializer(2, 'fan_in', 'random_normal',
                                                                    data_format=self.data_format)
        self.weight_regularizer = weight_regularizer
        self.trainable = trainable
        self.training = training
        self.levels = levels
        self.channels = channels

    def _conv2d(
            self,
            name,
            input,
            filters,
            kernel_size,
            strides=1,
            padding="SAME",
            dilations=1,
            bias=0
    ):
        # There are different shapes of weight metric between 'NCHW' and 'NHWC' mode
        if self.data_format == "NHWC":
            shape = (filters, kernel_size, kernel_size, input.shape[3])
        else:
            shape = (filters, input.shape[1], kernel_size, kernel_size)
        weight = flow.get_variable(
            name + "-weight",
            shape=shape,
            dtype=input.dtype,
            initializer=self.weight_initializer,
            regularizer=self.weight_regularizer,
            model_name="weight",
            trainable=self.trainable,
        )

        output = flow.nn.conv2d(input, weight, strides, padding, self.data_format, dilations, name=name + "_conv")

        # flow.watch(weight, watch_step)
        if bias != 0:
            bias_weight = flow.get_variable(
                name + "_bias",
                shape=(filters,),
                dtype=input.dtype,
                initializer=flow.constant_initializer(bias),
                regularizer=self.weight_regularizer,
            )
            output = flow.nn.bias_add(output, bias_weight, data_format=self.data_format)
        return output

    def _batch_norm(self, inputs, name=None):
        axis = 1
        if self.data_format == "NHWC":
            axis = 3
        return flow.layers.batch_normalization(inputs=inputs, axis=axis, name=name+"_bn")

    def _conv2d_transpose_layer(self,
                                name,  # name of layer
                                input,  # input of layer
                                kernel_size,  # kernel size of filters
                                in_channels,
                                out_channels,
                                strides=1,  # strides size
                                padding="SAME",  # padding is SAME or VALID
                                data_format="NCHW",  # N:batch size C: Number of channels H:height W:width
                                dilations=1,
                                trainable=False,  # trainable is True or False
                                use_bias=False,  # use_bias is True or False
                                bias_initializer=flow.zeros_initializer()  # flow.random_uniform_initializer(),
                                ):
        dilations = 1

        # weights in convolution layers
        weight = flow.get_variable(
            name + "-weight",
            shape=(in_channels, out_channels, kernel_size, kernel_size),
            dtype=flow.float,
            initializer=flow.variance_scaling_initializer(distribution="random_normal", data_format="NCHW"),
            regularizer=flow.regularizers.l2(0.0005),
            trainable=False,
        )

        out_shape = [input.shape[0], out_channels, input.shape[2] * strides, input.shape[3] * strides]
        output = flow.nn.conv2d_transpose(input, weight, strides=strides, output_shape=out_shape,
                                          dilations=dilations,
                                          padding=padding, data_format=data_format)  # deconvolution layer

        # bias in convolution layers
        if use_bias:
            bias = flow.get_variable(
                name + "-bias",
                shape=(out_channels,),
                dtype=input.dtype,
                initializer=bias_initializer,  # initialise bias
                regularizer=flow.regularizers.l2(0.0005)  # bias regularizer
            )
            # add bias if use_bias is true
            output = flow.nn.bias_add(output, bias, data_format)

        return output

    def base_layer(self, input):
        conv = self._conv2d("base_layer", input, self.channels[0], 7, 1)
        conv_bn = self._batch_norm(conv, "base_layer")
        conv_bn_relu = flow.nn.relu(conv_bn)
        return conv_bn_relu

    def _make_conv_level(self, level_name, x, planes, convs, stride=1, dilation=1):
        for i in range(convs):
            layer_name = "%s_%d" % (level_name, i)
            x = self._conv2d(layer_name, x, planes, 3, strides=stride if i == 0 else 1, dilations=dilation)
            x = self._batch_norm(x, layer_name)
            x = flow.nn.relu(x)
        return x

    def _block(self, name, x, inplanes, planes, stride=1, dilation=1, residual=None):
        if residual is None:
            residual = x

        out = self._conv2d(name+"_1", x, planes, kernel_size=3, strides=stride, dilations=dilation)
        out = self._batch_norm(out, name+"_1")
        out = flow.nn.relu(out)

        out = self._conv2d(name+"_2", out, planes, kernel_size=3, strides=1, dilations=dilation)
        out = self._batch_norm(out, name+"_2")

        out = flow.math.add(out, residual, name=name+"_block_neck")
        out = flow.nn.relu(out)

        return out

    def _Root(self, name, tree2, tree1, children, in_channels, out_channels, kernel_size, residual):
        if children.__len__() == 0:
            x = flow.concat([tree2, tree1], 1)
        else:
            x = flow.concat([tree2, tree1], 1)
            for i in range(children.__len__()):
                x = flow.concat([x, children[i]], 1)

        x = self._conv2d(name, x, out_channels, 1, 1)
        x = self._batch_norm(x, name=name)
        if residual:
            x = flow.math.add(x, children[0], name=name+"_root_neck")
        x = flow.nn.relu(x)

        return x

    def _Tree(self, name, x, levels, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1,
              dilation=1, root_residual=False, residual=None, children=None):

        children = [] if children is None else children
        if stride > 1:
            bottom = flow.nn.max_pool2d(x, stride, strides=stride, padding="SAME")
        else:
            bottom = x

        if in_channels != out_channels:
            residual = self._conv2d(name+"_res", bottom, out_channels, kernel_size=1, strides=1)
            residual = self._batch_norm(residual, name=name+"_res")
        else:
            residual = bottom

        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            children.append(bottom)
            root_dim += in_channels

        if levels == 1:
            tree1 = self._block(name+"_tree1", x, in_channels, out_channels, stride, dilation=dilation, residual=residual)
            tree2 = self._block(name+"_tree2", tree1, out_channels, out_channels, 1, dilation=dilation, residual=residual)
            out = self._Root(name+"_root", tree2, tree1, children, root_dim, out_channels, root_kernel_size, root_residual)
        else:
            tree1 = self._Tree(name+"_tree1", x, levels - 1, in_channels, out_channels, stride, root_dim=0,
                               root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            children.append(tree1)
            out = self._Tree(name+"_tree2", tree1, levels - 1, out_channels, out_channels, 1, root_dim=root_dim + out_channels,
                             root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual,
                             children=children)

        return out

    def dla(self, x, residual_root=False):

        y = []
        x = self.base_layer(x)

        x = self._make_conv_level("level0", x, self.channels[0], self.levels[0])
        y.append(x)
        x = self._make_conv_level("level1", x, self.channels[1], self.levels[1], stride=2)
        y.append(x)

        x = self._Tree("level2", x, self.levels[2], self.channels[1], self.channels[2], stride=2, level_root=False,
                       root_residual=residual_root)
        y.append(x)
        x = self._Tree("level3", x, self.levels[3], self.channels[2], self.channels[3], stride=2, level_root=True,
                       root_residual=residual_root)
        y.append(x)

        x = self._Tree("level4", x, self.levels[4], self.channels[3], self.channels[4], stride=2, level_root=True,
                       root_residual=residual_root)
        y.append(x)
        x = self._Tree("level5", x, self.levels[5], self.channels[4], self.channels[5], stride=2, level_root=True,
                       root_residual=residual_root)
        y.append(x)

        return y

    def backbone(self, x, residual_root=False):
        return self.dla(x)

    def proj(self, name, x, chi, out_dim):
        x = self._conv2d(name+"_proj", x, out_dim, 1, strides=1)
        x = self._batch_norm(x, name+"_proj")
        return flow.nn.relu(x)

    def node(self, name, x, chi, out_dim):
        x = self._conv2d(name+"_node", x, out_dim, kernel_size=3, strides=1)
        x = self._batch_norm(x, name+"_node")
        return flow.nn.relu(x)

    def IDA_UP(self, name, x, startp, endp, out_dim, channels, up_factors):
        for i in range(startp + 1, endp):
            name = "%s_%d" % (name, i)
            project = self.proj(name, x[i], channels[i - startp], out_dim)
            x[i] = self._conv2d_transpose_layer(name + "_transpose", project, int(up_factors[i - startp]) * 2,
                                                in_channels=out_dim, out_channels=out_dim,
                                                strides=int(up_factors[i - startp]))
            name_node_add = "%s%d%d" % (name+"_add", i, i-1)
            node_add = flow.math.add(x[i], x[i - 1], name=name_node_add)
            x[i] = self.node(name, node_add, out_dim, out_dim)

    def DLA_UP(self, x, first_level, channels, scales, in_channels=None):
        out = [x[-1]]
        scales = np.array(scales, dtype=int)
        if in_channels is None:
            in_channels = channels
        for i in range(len(x) - first_level - 1):
            j = -i - 2
            name = "%s_%d" % ("dla_up", i)
            self.IDA_UP(name, x, len(x) - i - 2, len(x), channels[j], in_channels[j:], scales[j:] // scales[j])
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]
            out.insert(0, x[-1])
        return out

    def _head(self, name, x, classes, head_conv):
        x = self._conv2d(name+"_head_1", x, head_conv, kernel_size=3, padding="SAME")
        x = flow.nn.relu(x)
        if name == "hm":
            x = self._conv2d(name + "_head_2", x, classes, kernel_size=1, strides=1, bias=-2.19)
        else:
            x = self._conv2d(name+"_head_2", x, classes, kernel_size=1, strides=1)

        return x

def DLA34(images, args, trainable=True, training=True):
    weight_regularizer = flow.regularizers.l2(args.wd) if args.wd > 0.0 and args.wd < 1.0 else None

    levels = [1, 1, 1, 2, 2, 1]
    channels = [16, 32, 64, 128, 256, 512]
    down_ratio = 4
    first_level = int(np.log2(down_ratio))
    last_level = 5
    scales = [2 ** i for i in range(len(channels[first_level:]))]
    out_channels = channels[first_level]
    builder = DLABuilder(levels, channels, weight_regularizer, trainable, training)

    with flow.scope.namespace("DLA34"):
        # flow.watch(images, watch_handler)
        backbone = builder.backbone(images)
        x = backbone[-1]

        x = flow.nn.avg_pool2d(x, 7, strides=1, padding="VALID")
        x = flow.layers.dense(inputs=flow.reshape(x, (x.shape[0], -1)),
                              units=1000, use_bias=True,
                              kernel_initializer=flow.random_normal_initializer(stddev=0.01),
                              bias_initializer=flow.zeros_initializer(), trainable=True)

    return x

def DLASeg(image, args, trainable=True, training=True):
    weight_regularizer = flow.regularizers.l2(0.00005)

    levels = [1, 1, 1, 2, 2, 1]
    channels = [16, 32, 64, 128, 256, 512]
    down_ratio = 4
    first_level = int(np.log2(down_ratio))
    last_level = 5
    scales = [2 ** i for i in range(len(channels[first_level:]))]
    out_channels = channels[first_level]
    builder = DLABuilder(levels, channels, weight_regularizer, trainable, training)

    with flow.scope.namespace("DLA34"):
        backbone = builder.backbone(image)
        dla_up = builder.DLA_UP(backbone, first_level, channels[first_level:], scales)

        y = []
        for i in range(last_level - first_level):
            y.append(dla_up[i])

        builder.IDA_UP("IDA_UP", y, 0, len(y), out_channels, channels[first_level:last_level],
                       [2 **i for i in range(last_level - first_level)])

        z = {}

        head_conv = 256

        z["hm"] = builder._head("hm", y[-1], 1, head_conv)
        z["wh"] = builder._head("wh", y[-1], 2, head_conv)
        z["id"] = builder._head("id", y[-1], 512, head_conv)
        z["reg"]= builder._head("reg", y[-1], 2, head_conv)

    return z

def CenterNet(image, args, trainable=True, training=True):
    weight_regularizer = flow.regularizers.l2(0.00005)

    levels = [1, 1, 1, 2, 2, 1]
    channels = [16, 32, 64, 128, 256, 512]
    down_ratio = 4
    first_level = int(np.log2(down_ratio))
    last_level = 5
    scales = [2 ** i for i in range(len(channels[first_level:]))]
    out_channels = channels[first_level]
    builder = DLABuilder(levels, channels, weight_regularizer, trainable, training)

    with flow.scope.namespace("DLA34"):
        backbone = builder.backbone(image)
        dla_up = builder.DLA_UP(backbone, first_level, channels[first_level:], scales)

        y = []
        for i in range(last_level - first_level):
            y.append(dla_up[i])

        builder.IDA_UP("IDA_UP", y, 0, len(y), out_channels, channels[first_level:last_level],
                       [2 **i for i in range(last_level - first_level)])

        z = {}

        head_conv = 256

        z["hm"] = builder._head("hm", y[-1], 80, head_conv)
        z["wh"] = builder._head("wh", y[-1], 2, head_conv)
        z["reg"]= builder._head("reg", y[-1], 2, head_conv)

    return z
