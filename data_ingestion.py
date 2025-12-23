def reading_data(file_path):
    with open(file_path , mode="r" , encoding= 'utf-8')  as f :
        text  = f.read()

    return text 


if __name__ == "__main__":
    # Example usage when run directly
    result = reading_data("data.txt")
    print(len(result))