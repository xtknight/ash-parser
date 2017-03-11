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
