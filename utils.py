import tensorflow as tf
from tensorflow.python.ops import control_flow_ops as cf

'''
Replace all digits with 9s like SyntaxNet
'''
def normalizeDigits(form):
    newform = ''
    for i in range(len(form)):
        if ord(form[i]) >= ord('0') and ord(form[i]) <= ord('9'):
            newform += '9'
        else:
            newform += form[i]
    return newform

'''
Gets array shape of dynamically shaped tensors

Ex.
tensorPrintShape(dense_golden, [dense_golden], 'dense_golden shape')
'''
def tensorPrintShape(inp, data, comment):
    def np_print(*args):
        for x in args:
            print(comment, x.shape)
    return cf.with_dependencies([tf.py_func(np_print, data, [])], inp)

'''
Ex.
tensorPrint(dense_golden, [dense_golden], 'dense_golden data')
'''
def tensorPrint(inp, data, comment):
    def np_print(*args):
        for x in args:
            print(comment, x)
    return cf.with_dependencies([tf.py_func(np_print, data, [])], inp)

tensorDumpValsCallCount = {}

'''
Ex.
tensorDumpVals(dense_golden, [dense_golden], '/tmp/ash_dense_golden_1', 1)
'''
# print only the desired_iter'th time the function is called (1-based)
# for this particular filename
def tensorDumpVals(inp, data, fname, desired_iter):
    global tensorDumpValsCallCount

    def np_print(*args):
        global tensorDumpValsCallCount

        if fname not in tensorDumpValsCallCount:
            tensorDumpValsCallCount[fname] = 0
        tensorDumpValsCallCount[fname] += 1

        # only execute for the iteration # desired
        if tensorDumpValsCallCount[fname] == desired_iter:
            fd = open(fname, 'w')

            for x in args:
                for elem in x.flatten():
                    fd.write('%.8f\n' % elem)

            fd.close()

    return cf.with_dependencies([tf.py_func(np_print, data, [])], inp)

'''
Ex.
tensorDumpValsAllIter(dense_golden, [dense_golden], '/tmp/ash_dense_golden')
'''

def tensorDumpValsAllIter(inp, data, fname):
    global tensorDumpValsCallCount

    def np_print(*args):
        global tensorDumpValsCallCount

        if fname not in tensorDumpValsCallCount:
            tensorDumpValsCallCount[fname] = 0
        tensorDumpValsCallCount[fname] += 1

        fd = open('%s_%04d' % (fname, tensorDumpValsCallCount[fname]), 'w')

        for x in args:
            for elem in x.flatten():
                fd.write('%.8f\n' % elem)

        fd.close()

    return cf.with_dependencies([tf.py_func(np_print, data, [])], inp)

