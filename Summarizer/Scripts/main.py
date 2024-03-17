from summarizer import summarizer
def main():
#define for loop logic here to iterate over Resources
    filename = input("Please enter the filename without .txt ext: ")
    summarizer(filename)

if __name__ == '__main__':
    main()