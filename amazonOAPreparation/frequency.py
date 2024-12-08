def count_char_frequency(charArray):
    # Using a dictionary to store frequencies
    freq_dict = {}

    # Go through each character in the array
    for char in charArray:
        # Check if character is already in the dictionary
        if char in freq_dict:
            freq_dict[char] += 1
        else:
            freq_dict[char] = 1

    # Display the characters and their frequencies in order
    for char, freq in freq_dict.items():
        print(f"'{char}': {freq}")

# Example
charArray = ['a', 'b', 'a', 'c', 'b', 'a', 'd', 'e', 'f', 'e']
count_char_frequency(charArray)