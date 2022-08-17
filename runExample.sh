FILE=exampleData.json

if [ -f "$FILE" ]; then
    printf '\n\nExample data available\n\n'
else 
    printf '\n\nGetting some example data\n\n'
    wget -O- http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Automotive_5.json.gz | gunzip -c > exampleData.json
fi

python3 -W ignore::DeprecationWarning modelExample.py
open topicsVis.html
