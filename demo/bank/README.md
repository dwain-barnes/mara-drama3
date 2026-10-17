Mara's mood clips, one per label the chat LLM can put in front of a sentence.
Each is a line rendered from the p283 reference with a direction such as
`[excited]`, then encoded back to Drama 3's codec codes: `<mood>.codes` is the
clip as int32 codes, ten per frame, and `<mood>.txt` is what it says. The
server hands Drama 3 the recording first and the mood clip second, so the
voice stays the same and only the delivery changes. `make_bank.py` renders
them again if the voice changes; the wavs it writes are not tracked.
