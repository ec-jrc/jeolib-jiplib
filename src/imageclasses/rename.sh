cat rename.txt | awk -F ' ' '{ print $3 }' | sed 's/(.*//' | sed 's/\**//' | sed 's/$/\t\t/' > $1/c_names.txt
cat rename.txt | sed 's/^.* //g' > $1/methods_names.txt
paste  $1/c_names.txt $1/methods_names.txt > $1/old2NewNames.txt
nlines=`cat  $1/old2NewNames.txt | awk 'NF' | wc -l` 
cat  $1/old2NewNames.txt | awk 'NF' | sed 's/\t\t\t/\t/' | awk -v nlin=$nlines -F '\t' 'BEGIN { print "{" } { if (NR < nlin) print "\""$1"\":\t\t\""$2"\"," ; else print "\""$1"\":\t\t\""$2"\""  } END { print "}" }' >  $1/old2NewNames.json
