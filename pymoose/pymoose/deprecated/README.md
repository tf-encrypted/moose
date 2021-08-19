# Prototype Runtime - Deprecated

This directory contains Python modules related to the initial prototype of the compiler & runtime. 
This code was in use for our initial POC implementations, and has been useful to keep as reference during the migration to Rust.
It's also been useful to cite during development/feature discussions since then, which is why it's been kept around so far.
So don't expect this code to work forever -- WYSIWYG.

Future python code may move here if there is code in the Rust codebase that achieves roughly equivalent functionality.
This will be the case for the Python compiler code once all passes have been migrated to the Rust compiler (elk_compiler there).

The structure of this directory matches the directory of the original Python prototype.
Some files may be mirrored in the non-deprecated Python codebase, which means that some part of that file was deprecated and moved into this folder.

When deprecating code from the main Python codebase, please make sure you are not breaking consistency with the deprecated codebase. For example, one should make sure paths & import statements are updated in the deprecated codebase, whenever applicable.
In general, full backwards compatibility with code in the main Python codebase is not necessary.
Internal consistency within deprecated folders, however, _is necessary_ in order to successfully deprecate the code.
