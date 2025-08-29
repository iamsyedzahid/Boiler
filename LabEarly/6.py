countries_population = {
    "China": 1393409038,
    "India": 1366400000,
    "USA": 331893745,
    "Indonesia": 273523615,
    "Pakistan": 220892340
}

first_highest = ("", 0)
second_highest = ("", 0)
third_highest = ("", 0)

for country in countries_population:
    population = countries_population[country]
    
    if population > first_highest[1]:
        third_highest = second_highest
        second_highest = first_highest
        first_highest = (country, population)
    elif population > second_highest[1]:
        third_highest = second_highest
        second_highest = (country, population)
    elif population > third_highest[1]:
        third_highest = (country, population)

print("Top 3 most populated countries:")
print(first_highest)
print(second_highest)
print(third_highest)