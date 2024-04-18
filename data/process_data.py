import csv

data = []
questions, answers = [], []
with open("/Users/chihuahuaiscute/Desktop/cshw/PFFT/data/MedQA_new_rewrite.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row['text'].strip())

num_fail_data = 0
collected_data = []
for i in range(len(data)):
    idx_q = data[i].find("Question:")
    idx_a = data[i].find("A.")
    idx_b = data[i].find("B.")
    idx_c = data[i].find("C.")
    idx_d = data[i].find("D.")
    idx_answer = data[i].find('Answer:')
    question = data[i][idx_q:idx_a].replace('\n', '').strip()
    option_a = data[i][idx_a:idx_b].replace('\n', '').strip()
    option_b = data[i][idx_b:idx_c].replace('\n', '').strip()
    option_c = data[i][idx_c:idx_d].replace('\n', '').strip()
    option_d = data[i][idx_d:idx_answer].replace('\n', '').strip()
    answer = data[i][idx_answer:]

    if idx_answer == -1 or idx_q == -1 :
        num_fail_data += 1
        pass
    elif ":[opt" in data[i]:
        pass
    else: 
        tmp = f"{question}\n{option_a}\n{option_b}\n{option_c}\n{option_d}\n{answer}"
        collected_data.append(tmp)
        _ = tmp.split('\n')[-1]
        # if "Answer" not in _ :
            
        # import pdb
        # pdb.set_trace()
print(num_fail_data)

import pandas as pd
df = pd.DataFrame(columns=['text'], data=collected_data)
df.to_csv('test.csv')
# for i in range(len(data)) :
#     idx_q = data[i].find("Question:")
#     idx_a = data[i].find("Answer:")

#     questions.append(data[i][idx_q + len("Question:") : idx_a].strip())
#     answers.append(data[i][idx_a + len("Answer:") :].strip())

# to_write = []


# import pandas as pd
# for i in range(len(data)):
#     to_write.append([questions[i], answers[i]])

# print(len(questions), len(answers))
# df = pd.DataFrame(columns=['Question', 'Answer'], data=to_write)
# df.to_csv('MedQA_1000_reformat.csv')

