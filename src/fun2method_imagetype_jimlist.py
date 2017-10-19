# first 20170830 by pieter.kempeneers@ec.europa.eu


def fun2method(inputfile, outputfile_basename):
    """converts MIALib C function declarations into JIPLib C++ methods (outputfile_basename.cc file) and C++ method declarations (outputfile_basename.h file).  Currently only convert desctuctive functions, i.e., ERROR_TYPE functions, with IMAGE ** as first argument

    :param inputfile: string for input file containing extern declarations
    :param outputfile_basename: string for output file basename (i.e. without extension)
    :returns: True on success, False otherwise
    :rtype:

    """

    import re
    import json
    old2newDic = json.load(open("old2NewNames.json"))

    ifp=open(inputfile, 'r')

    f = open(outputfile_basename+'.cc', 'w')
    fh = open(outputfile_basename+'.h', 'w')


    lines=ifp.readlines()

    for line in lines:
        print(line)

        name=re.match(r'extern IMAGE \*(.*)\((.*)\);',line)
        args=re.split(',', name.group(2))
        re.sub('\**', '', re.sub('.* ', '', args[0]))

        arglist = []

        for i in args:
            atype=re.match(' *(.* \**)', i).group(1)
            aname=re.sub('\**', '', re.sub('.* ', '', i))
            arglist.append([atype, aname])


        a= { "name" : name.group(1),
             "arguments" : arglist }

        print(a)

        imDeclare=[]
        imRasterArray=[]

        GTDeclare=[]
        GTVars=[]

        GDTypes = ['GDT_Byte', 'GDT_Int16', 'GDT_UInt16', 'GDT_Int32', 'GDT_UInt32', 'GDT_Float32', 'GDT_Float64']
        MIATypes = ['uc_', 's_', 'us_', 'i32_', 'u32_', 'f_', 'd_']
        CTypes = ['unsigned char', 'short int', 'unsigned short int', 'int', 'unsigned int', 'float', 'double']

        methodDeclaration='std::shared_ptr<Jim> JimList::'+old2newDic.get(a.get("name"))+'('
        print(methodDeclaration)

        cSeparator=', '
        separator = '' # default value in case there are no arguments besides the input image

        if a.get("arguments")[0][0] != 'IMAGE **':
            print('WARNING: ',line,' does not have IMAGE ** as first argument')
            continue
        else:
            imDeclare.append(a.get("arguments")[0][0]+' '+a.get("arguments")[0][1]+';')
            cCall=a.get("arguments")[0][1]

        imCount=1
        setNc=False
        for idx, arg in enumerate(a.get("arguments")[1:len(a.get("arguments"))]):
            print(methodDeclaration)
            print(idx)
            print(idx)
            print(separator)
            print(arg)
            print(arg[0])

            if (arg[0]=='IMAGE *'):
                methodDeclaration+=separator+'Jim& imRaster_'+arg[1]
                imRasterArray.append('imRaster_'+arg[1])
                imDeclare.append(arg[0]+' '+arg[1]+'=imRaster_'+arg[1]+'.getMIA();')
                separator=', '
            elif (arg[0]=='G_TYPE '): # G_TYPE as double in python
                GTDeclare.append(arg[0]+' '+arg[1]+';')
                GTVars.append(arg[1])
                methodDeclaration+=separator+'double '+'d_'+arg[1]
                separator=', '
            elif (arg[1]=='nc'):
                setNc=True
                separator=''
            else:
                methodDeclaration+=separator+arg[0]+' '+arg[1]
                separator=', '

            cCall+=cSeparator+arg[1]

        fh.write(re.sub(r'JimList::','',methodDeclaration+');\n'))

        methodDeclaration+=')'

        f.write(methodDeclaration+'{')
        f.write('\n\ttry{')
        #f.write('\n\t\t'+a.get("arguments")[0][0]+' '+a.get("arguments")[0][1]+' = 0;') # assigned later depending on destructive or not
        f.write('\n\t\t'+'IMAGE * imout = 0;')
        llen=len(imDeclare)
        for i in range(0,llen):
            f.write('\n\t\t'+imDeclare[i])
        for i in GTDeclare:
            f.write('\n\t\t'+i)

        if len(GTDeclare)>0:
            f.write('\n\t\tswitch(getDataType()){')
            for idx, GDType in enumerate(GDTypes):
                f.write('\n\t\tcase('+GDType+'):')
                for var_idx, GTVar in enumerate(GTVars):
                    f.write('\n\t\t\t'+GTVar+'.'+MIATypes[idx]+'val=static_cast<'+CTypes[idx]+'>(d_'+GTVars[var_idx]+');')
                    if (GDType=='GDT_Byte'):
                        f.write('\n\t\t\t'+GTVar+'.'+'generic_'+'val=static_cast<'+CTypes[idx]+'>(d_'+GTVars[var_idx]+');')
                f.write('\n\t\t\tbreak;')
            f.write('''
    \t\tdefault:
    \t\t\tstd::string errorString="Error: data type not supported";
    \t\t\tthrow(errorString);
    \t\t\tbreak;
            \t\t}''')

        f.write('\n\t\t'+a.get("arguments")[0][1]+' = (IMAGE **) malloc(this->size()*sizeof(IMAGE **));')
        f.write('\n\t\tfor(int iimg=0;iimg<this->size();++iimg)')
        f.write('\n\t\t\t'+a.get("arguments")[0][1]+'[iimg]=this->getImage(iimg)->getMIA();')

        if setNc:
            f.write('\n\t\tint nc = this->size();')
        f.write('\n\t\timout =::'+a.get("name")+'('+cCall+');')
        f.write('\n\t\tif (imout){')
        f.write('\n\t\t\tstd::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout);')
        f.write('\n\t\t\timgWriter->copyGeoTransform(*front());')
        f.write('\n\t\t\timgWriter->setProjection(front()->getProjectionRef());')
        f.write('\n\t\t\treturn(imgWriter);')
        f.write('\n\t\t}')

        f.write('\n\t\telse{')
        f.write('\n\t\t\tstd::string errorString="Error: '+a.get("name")+'() function in MIA failed, returning NULL pointer";')
        f.write('\n\t\t\tthrow(errorString);')
        f.write('\n\t\t}')
        f.write('\n\t}')

        f.write('\n\tcatch(std::string errorString){')
        f.write('\n\t\tstd::cerr << errorString << std::endl;')
        f.write('\n\t\treturn(0);')
        f.write('\n\t}')
        f.write('\n\tcatch(...){')
        f.write('\n\t\treturn(0);')
        f.write('\n\t}')
        f.write('\n}')
        f.write('\n')

    ifp.close()

    f.close()
    fh.close()

    return(True)



####################################################################

import sys, getopt

def main(argv):
   inputfile="mialib_imagetype"
   outputfile="fun2method_imagetype_jimlist"
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('fun2method.py -i <inputfile> -o <outputfilebasename>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('fun2method.py -i <inputfile> -o <outputfilebasename>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print('Input file is "', inputfile)
   print('Output file is "', outputfile)

   fun2method(inputfile, outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])


# cat /usr/local/include/mialib/mialib_*.h | grep '^extern IMAGE \*[^\*]'  > mialib_image_type
# python fun2method_imagetype.py  -i mialib_image_type -o fun2method_imagetype
# to automatically insert content of fun2method in jim.h within placeholder //start insert from fun2method -> //end insert from fun2method
# sed -i -ne '/\/\/start insert from fun2method_imagetype/ {p; r fun2method_imagetype.h' -e ':a; n; /\/\/end insert from fun2method_imagetype/ {p; b}; ba}; p' jim.h
