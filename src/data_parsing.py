import json 
import pandas

graph = {}
table = {}
coo = [[], []]

with open("DBLPOnlyCitationOct19.txt") as file:
    file.readline()
    curr = {}
    cite = []
    index = None
    while line := file.readline():
        if len(line) == 1 and index:
            graph[index] = cite
            table[index] = curr
            for i in cite:
                coo[0].append(int(index))
                coo[1].append(int(i))
            index = None
            curr = {}
            cite = []

        elif line[:2] == "#*":
            curr["title"] = line[2:-1]
        elif line[:2] == "#@":
            curr["authors"] = line[2:-1]
        elif line[:2] == "#t":
            curr["year"] = int(line[2:-1])
        elif line[:2] == "#c":
            curr["venue"] = line[2:-1]
        elif line[:2] == "#!":
            curr["abstract"] = line[2:-1]
        elif line[:2] == "#%":
            cite.append(line[2:-1])
        elif line[:6] == "#index":
            index = line[6:-1]

print("processing done")
pandas.DataFrame.from_dict(table, orient="index").to_csv("table.csv")
print("table done")

with open('graph.json', 'w') as f:
    json.dump(graph, f)
with open('coo.json', 'w') as f:
    json.dump(coo, f)
print("graph done")