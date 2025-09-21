import random


def tournament_selection(numbers, tournament_size, output_file):
    selected_numbers = []
    for _ in range(len(numbers)):
        tournament = random.sample(numbers, tournament_size)
        winner = min(tournament)
        winner_index = numbers.index(winner)+1
        selected_numbers.append(winner_index)
    with open(output_file, "w") as file:
        for number in selected_numbers:
            file.write(str(number) + "\n")

    return selected_numbers


def main():
    numbers = []
    with open(r'C:\Users\22279\Documents\Test2_alexnet\accuracy\accuracy1.txt', 'r') as file:
        for line in file:
            number = line.strip()
            numbers.append(float(number))

    tournament_size = 60
    output_file = "Select_list.txt"
    selected_numbers=tournament_selection(numbers, tournament_size, output_file)


if __name__ == '__main__':
    main()
