# -*- coding: utf-8 -*-


from myapi.app import create_app, load_model, process_args
import uvicorn

import pdb

def main():
    args = process_args()
    model, tokenizer, device, prompt_template = load_model(args)
    app = create_app(model, tokenizer, device, prompt_template)
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)
    print("Visit http://localhost:8000/docs for API document.")

if __name__ == "__main__":
    main()
