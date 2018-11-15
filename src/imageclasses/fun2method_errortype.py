# first 20161111 by Pierre.Soille@jrc.ec.europa.eu

def fun2method(inputfile, outputfile_basename):
    """converts MIALib C function declarations into JIPLib C++ methods (outputfile_basename.cc file) and C++ method declarations (outputfile_basename.h file).  Currently only convert desctuctive functions, i.e., ERROR_TYPE functions, with IMAGE * as first argument (IMAGE ** not yet taken into account).

    :param inputfile: string for input file containing extern declarations
    :param outputfile_basename: string for output file basename (i.e. without extension)
    :returns: True on success, False otherwise
    :rtype:

    """

    import re
    import json
    # for writing a dictionary in json file
    #j son.dump(old2NewDict, open("text.txt",'w'))
    # reading dictionary in json file
    old2newDic = json.load(open("old2NewNames.json"))

    ifp=open(inputfile, 'r')

    f = open(outputfile_basename+'.cc', 'w')
    fh = open(outputfile_basename+'.h', 'w')


    lines=ifp.readlines()

    for line in lines:
        print(line)

        name=re.match(r'extern ERROR_TYPE (.*)\((.*)\);',line)
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

        methodDeclaration='std::shared_ptr<Jim> Jim::'+old2newDic.get(a.get("name"))+'('
        print(methodDeclaration)

        cSeparator=', '
        separator = '' # default value in case there are no arguments besides the input image

        if a.get("arguments")[0][0] != 'IMAGE *':
            print('WARNING: ',line,' does not have IMAGE * as first argument')
            continue
        else:
            imDeclare.append(a.get("arguments")[0][0]+' '+a.get("arguments")[0][1]+'=this->getMIA(iband);')
            cCall=a.get("arguments")[0][1]

        imCount=1
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
                imDeclare.append(arg[0]+' '+arg[1]+'=imRaster_'+arg[1]+'.getMIA(iband);')
            elif (arg[0]=='G_TYPE '): # G_TYPE as double in python
                GTDeclare.append(arg[0]+' '+arg[1]+';')
                GTVars.append(arg[1])
                methodDeclaration+=separator+'double '+'d_'+arg[1]
            else:
                methodDeclaration+=separator+arg[0]+' '+arg[1]

            cCall+=cSeparator+arg[1]
            separator=', '

        # fh.write(re.sub(r'Jim::','',methodDeclaration+separator+'int iband=0'+cSeparator+'bool destructive=false);\n'))
        fh.write(re.sub(r'Jim::','',methodDeclaration+separator+'int band=-1);\n'))

        # methodDeclaration+=separator+'int iband'+cSeparator+'bool destructive)'
        methodDeclaration+=separator+'int band)'

        f.write(methodDeclaration+'{')
        f.write('\n\ttry{')
        f.write('\n\t\tif(nrOfBand()<=band){')
        f.write('\n\t\t\tstd::string errorString=\"Error: band number exceeds number of bands in input image\";')
        f.write('\n\t\t\tthrow(errorString);')
        f.write('\n\t\t}')
        for i in imRasterArray:
            f.write('\n\t\tif('+i+'.nrOfBand()<=band){')
            f.write('\n\t\t\tstd::string errorString=\"Error: band number exceeds number of bands in input image\";')
            f.write('\n\t\t\tthrow(errorString);')
            f.write('\n\t\t}')

        f.write('\n\t\t//make a copy of this')
        f.write('\n\t\tstd::shared_ptr<Jim> copyImg=this->clone();')
        f.write('\n\t\tstd::vector<unsigned int> bands;')
        f.write('\n\t\tif(band<0){')
        f.write('\n\t\t\tfor(unsigned int iband=0;iband<nrOfBand();++iband)')
        f.write('\n\t\t\t\tbands.push_back(iband);')
        f.write('\n\t\t}')
        f.write('\n\t\telse')
        f.write('\n\t\t\tbands.push_back(band);')
        f.write('\n\t\tfor(std::vector<unsigned int>::const_iterator bit=bands.begin();bit!=bands.end();++bit){')
        f.write('\n\t\t\tunsigned int iband=*bit;')

        f.write('\n\t\t\t'+a.get("arguments")[0][0]+' '+a.get("arguments")[0][1]+' = 0;') # assigned later depending on destructive or not
        llen=len(imDeclare)
        for i in range(1,llen):  # skip first argument since it depends on whether destructive or not
            f.write('\n\t\t\t'+imDeclare[i])
        for i in GTDeclare:
            f.write('\n\t\t\t'+i)

        if len(GTDeclare)>0:
            f.write('\n\t\t\tswitch(getDataType()){')
            for idx, GDType in enumerate(GDTypes):
                f.write('\n\t\t\tcase('+GDType+'):')
                for var_idx, GTVar in enumerate(GTVars):
                    f.write('\n\t\t\t\t'+GTVar+'.'+MIATypes[idx]+'val=static_cast<'+CTypes[idx]+'>(d_'+GTVars[var_idx]+');')
                    if (GDType=='GDT_Byte'):
                        f.write('\n\t\t\t\t'+GTVar+'.'+'generic_'+'val=static_cast<'+CTypes[idx]+'>(d_'+GTVars[var_idx]+');')
                f.write('\n\t\t\t\tbreak;')
            f.write('''
    \t\t\tdefault:
    \t\t\t\tstd::string errorString="Error: data type not supported";
    \t\t\t\tthrow(errorString);
    \t\t\t\tbreak;
            \t\t\t}''')
        f.write('\n\t\t\t'+a.get("arguments")[0][1]+'=copyImg->getMIA(iband);')

        f.write('\n\t\t\tif(::'+a.get("name")+'('+cCall+') == NO_ERROR){')
        f.write('\n\t\t\t\tcopyImg->setMIA(iband);')
        for i in imRasterArray:
          f.write('\n\t\t\t\t'+i+'.setMIA(iband);')
        f.write('\n\t\t\t\treturn(copyImg);')
        f.write('\n\t\t\t}')

        f.write('\n\t\t\telse{')
        f.write('\n\t\t\t\tcopyImg->setMIA(iband);')
        for i in imRasterArray:
           f.write('\n\t\t\t\t'+i+'.setMIA(iband);')
        f.write('\n\t\t\t\tstd::string errorString="Error: '+a.get("name")+'() function in MIA failed, returning NULL pointer";')
        f.write('\n\t\t\t\tthrow(errorString);')
        f.write('\n\t\t\t}')

        f.write('\n\t\t}')
        f.write('\n\t}')

        f.write('\n\tcatch(std::string errorString){')
        f.write('\n\t\tstd::cerr << errorString << std::endl;')
        f.write('\n\tthrow;')
        f.write('\n\t}')
        f.write('\n\tcatch(...){')
        f.write('\n\t\tthrow;')
        f.write('\t\t\n}')
        f.write('\n}\n')
#         f.write('''
#         catch(std::string errorString){
#         \tstd::cerr << errorString << std::endl;
#         throw;
#         }
#         catch(...){
#         throw;
#         }
# }\n''')

    ifp.close()

    f.close()
    fh.close()

    return(True)



####################################################################

import sys, getopt

def main(argv):
   inputfile="mialib_errortype"
   outputfile="fun2method_errortype"
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




# cat /home/soillpi/workstation/jip/mia//core/c/mialib_*.h | grep '^extern ERROR'  > mialib_error_type
# cat /home/soillpi/work/jip20170201/mia//core/c/mialib_*.h | grep '^extern ERROR'  > mialib_error_type
# python fun2method_errortype.py  -i mialib_error_type -o fun2method_errortype
# to automatically insert content of fun2method in jim.h within placeholder //start insert from fun2method -> //end insert from fun2method
# sed -i -ne '/\/\/start insert from fun2method_errortype/ {p; r fun2method_errortype.h' -e ':a; n; /\/\/end insert from fun2method_errortype/ {p; b}; ba}; p' jim.h
