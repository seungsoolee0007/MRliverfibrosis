import base64

def FileToBase64(filename):
    f = open(filename, "rb")
    c = f.read()
    print("raw file length : %s" % (len(c)))
    return base64.b64encode(c).decode("utf-8")

def Base64ToFile(filename, encoded):
    print("whatthe")
    f = open(filename, "wb")
    print("decoding file length : %s" % (len(encoded)))
    f.write(base64.b64decode(encoded))    

    
