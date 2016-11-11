
a={
    "name" : "thresh",
    "arguments" : [ ['IMAGE *', 'im'],
                    ['G_TYPE', 'gt1',],
                    ['G_TYPE', 'gt2',],
                    ['G_TYPE', 'gbg',],
                    ['G_TYPE', 'gfg',]
                    ]
    }




def fun2method(a):
    imDeclare=[]
    imRasterArray=[]

    GTDeclare=[]
    GTVars=[]

    GDTypes = ['GDT_Byte', 'GDT_Int16', 'GDT_UInt16', 'GDT_Int32', 'GDT_UInt32', 'GDT_Float32', 'GDT_Float64']
    MIATypes = ['uc_', 's_', 'us_', 'i32_', 'u32_', 'f_', 'd_']
    CTypes = ['unsigned char', 'short int', 'unsigned short int', 'int', 'unsigned int', 'float', 'double']

    methodDeclaration='CPLErr Jim::'+a.get("name")+'('

    cSeparator=', '

    if a.get("arguments")[0][0] != 'IMAGE *':
        print 'error'
    else:
        imDeclare.append(a.get("arguments")[0][0]+' '+a.get("arguments")[0][1]+'=this->getMIA(iband);')
        cCall=a.get("arguments")[0][1]

    imCount=1
    for idx, arg in enumerate(a.get("arguments")[1:len(a.get("arguments"))]):
        print idx
        print arg
        if idx==0:
            separator=''
        else:
            separator=', '

        if (arg[0]=='IMAGE *'):
            methodDeclaration+=separator+'Jim& imRaster_'+arg[1]
            print methodDeclaration
            imRasterArray.append('imRaster_'+arg[1])
            imDeclare.append(arg[0]+' '+arg[1]+'=imRaster_'+arg[1]+'.getMIA(iband);')
        elif (arg[0]=='G_TYPE'): # G_TYPE as double in python
            GTDeclare.append(arg[0]+' '+arg[1]+';')
            GTVars.append(arg[1])
            methodDeclaration+=separator+'double '+'d_'+arg[1]
        else:
            methodDeclaration+=separator+arg[0]+' '+arg[1]

        cCall+=cSeparator+arg[1]


    methodDeclaration+=cSeparator+'int iband){'


    f = open('workfile', 'w')

    f.write(methodDeclaration)
    f.write('\n\t try{')
    f.write('\n\t\t if(nrOfBand()<=iband){')
    f.write('\n\t\t\t std::string errorString=\"Error: band number exceeds number of bands in input image\";')
    f.write('\n\t\t\t throw(errorString);')
    f.write('\n\t\t }')
    for i in imRasterArray:
        f.write('\n\t\t if('+i+'.nrOfBand()<=iband){')
        f.write('\n\t\t\t std::string errorString=\"Error: band number exceeds number of bands in input image\";')
        f.write('\n\t\t\t throw(errorString);')
        f.write('\n\t\t }')
    for i in imDeclare:
        f.write('\n\t\t'+i)
    for i in GTDeclare:
        f.write('\n\t\t'+i)

    if len(GTDeclare)>0:
        f.write('\n\t\tswitch(getDataType()){')
        for idx, GDType in enumerate(GDTypes):
            f.write('\n\t\tcase('+GDType+'):')
            for var_idx, GTVar in enumerate(GTVars):
                f.write('\n\t\t\t'+GTVar+'.'+MIATypes[idx]+'val=static_cast<'+CTypes[idx]+'>(d_'+GTVars[var_idx]+');')
            f.write('\n\t\t\tbreak;')
        f.write('''
\t\tdefault:
\t\t\tstd::string errorString="Error: data type not supported";
\t\t\tthrow(errorString);
\t\t\tbreak;
\t\t}''')
                    
        

    f.write('\n\t\tif(::'+a.get("name")+'('+cCall+') == NO_ERROR){')
    f.write('\n\t\t\tthis->setMIA(iband);')
    for i in imRasterArray:
        f.write('\n\t\t\t'+i+'.setMIA(iband);')
    f.write('\n\t\t\treturn(CE_None);')
    f.write('\n\t\t}')
    f.write('\n\t\telse{')
    f.write('\n\t\t\tthis->setMIA(iband);')
    for i in imRasterArray:
        f.write('\n\t\t\t'+i+'.setMIA(iband);')
    f.write('\n\t\t\tstd::string errorString="Error: arith function in MIA failed";')
    f.write('\n\t\t\tthrow(errorString);')
    f.write('\n\t\t}')
    f.write('\n\t}')
    f.write('''
    \tcatch(std::string errorString){
    \t\tstd::cerr << errorString << std::endl;
    \t\treturn(CE_Failure);
    \t}
    \tcatch(...){
    \t\treturn(CE_Failure);
    \t}
    }
''')

    f.close()

