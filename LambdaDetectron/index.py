from app1 import parse_args, main

def handler(event, context):
    print(event)

    args = parse_args()
    args.opts+=["OUTPUT_DIR", '/tmp/output']
    print("New2")
    if 'index.handler' in args.opts:
        args.opts.remove('index.handler')

    main(args, event)
    return "done"

if __name__ == "__main__":
	handler('', '')