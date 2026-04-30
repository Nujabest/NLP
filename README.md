recup données : 

git clone https://gitlab.teklia.com/ckermorvant/arkindex_archelec

déziper données : 

cd arkindex_archelec && shopt -s globstar && for f in text_files/**/*.zip; do unzip -q -d "$(dirname "$f")" "$f"; done


