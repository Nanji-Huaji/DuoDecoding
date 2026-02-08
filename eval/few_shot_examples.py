# 官方 GSM8K CoT few-shot 示例（使用标准格式）
GSM8K_FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. #### 29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8"
    }
]

SUMMARIZATION_FEW_SHOT_EXAMPLES = [
    {
        "article": "The world's largest volcanic eruption over the past century occurred in June 1991 in the Philippines, when Mount Pinatubo erupted after hundreds of years of dormancy. The eruption produced a massive cloud of ash and gas that reached 22 miles into the atmosphere, causing global temperatures to drop by 1 degree Fahrenheit for several years. More than 800 people were killed and 10,000 made homeless.",
        "summary": "Mount Pinatubo erupted in 1991 in the Philippines, killing 800 people and lowering global temperatures."
    },
    {
        "article": "SpaceX successfully landed its Falcon 9 rocket on a drone ship at sea on Friday after launching a satellite into orbit. This marks the third successful landing on a drone ship for Elon Musk's space exploration company. The achievement is a significant step towards SpaceX's goal of reusing rockets to lower the cost of space travel.",
        "summary": "SpaceX landed a Falcon 9 rocket on a drone ship for the third time, advancing rocket reuse goals."
    },
    {
        "article": "Researchers at Stanford University have developed a new type of battery that can charge in less than a minute. The aluminum-ion battery is safer than current lithium-ion batteries and could be used in smartphones and other electronic devices. While it currently doesn't hold as much charge as lithium-ion batteries, the researchers are working to improve its storage capacity.",
        "summary": "Stanford researchers developed a fast-charging, safe aluminum-ion battery for electronics."
    }
]

HUMANEVAL_FEW_SHOT_EXAMPLES = [
    {
        "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"",
        "completion": "    return a + b"
    },
    {
        "prompt": "def concatenate(strings: List[str]) -> str:\n    \"\"\" Concatenate list of strings into a single string\n    >>> concatenate([])\n    ''\n    >>> concatenate(['a', 'b', 'c'])\n    'abc'\n    \"\"\"",
        "completion": "    return ''.join(strings)"
    },
    {
        "prompt": "def is_palindrome(string: str) -> bool:\n    \"\"\"\n    Check if given string is palindrome\n    >>> is_palindrome('')\n    True\n    >>> is_palindrome('aba')\n    True\n    >>> is_palindrome('aaaaa')\n    True\n    >>> is_palindrome('zbcd')\n    False\n    \"\"\"",
        "completion": "    return string == string[::-1]"
    }
]

TRANSLATION_FEW_SHOT_EXAMPLES = [
    {
        "german": "Guten Morgen, wie geht es Ihnen heute?",
        "english": "Good morning, how are you today?"
    },
    {
        "german": "Die Sonne scheint und die Vögel singen.",
        "english": "The sun is shining and the birds are singing."
    },
    {
        "german": "Ich lerne Deutsch, um in Deutschland zu arbeiten.",
        "english": "I am learning German to work in Germany."
    }
]

def get_few_shot_prompt(task, num_shots):
    if num_shots <= 0:
        return ""
    
    prompt = ""
    if task == "gsm8k":
        for i in range(min(num_shots, len(GSM8K_FEW_SHOT_EXAMPLES))):
            example = GSM8K_FEW_SHOT_EXAMPLES[i]
            prompt += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
    elif task in ["cnndm", "xsum", "summarization"]:
        for i in range(min(num_shots, len(SUMMARIZATION_FEW_SHOT_EXAMPLES))):
            example = SUMMARIZATION_FEW_SHOT_EXAMPLES[i]
            prompt += f"Article: {example['article']}\nSummary: {example['summary']}\n\n"
    elif task == "humaneval":
        for i in range(min(num_shots, len(HUMANEVAL_FEW_SHOT_EXAMPLES))):
            example = HUMANEVAL_FEW_SHOT_EXAMPLES[i]
            prompt += f"{example['prompt']}\n{example['completion']}\n\n"
    elif task == "translation":
        for i in range(min(num_shots, len(TRANSLATION_FEW_SHOT_EXAMPLES))):
            example = TRANSLATION_FEW_SHOT_EXAMPLES[i]
            prompt += f"German: {example['german']}\nEnglish: {example['english']}\n\n"
    
    return prompt
