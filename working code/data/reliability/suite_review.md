# Reliability suite - examples

## R1_repeatability (50 variants)

- `Q-T09-08~rep1`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point?
- `Q-T09-08~rep2`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point?
- `Q-T09-08~rep3`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point?
- `Q-T09-08~rep4`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point?
- `Q-T09-08~rep5`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point?
- `Q-T03-05~rep1`: When comparing the weekly working-time rules, what legal framework does each handbook use, and what weekly limit or rule is described?

## R1_semantic (90 variants)

- `Q-T09-03~para1`: A manager wants to know whether both handbooks require staff to keep company and client information confidential during and after employment. What confidentiality rule do both handbooks share?
- `Q-T09-03~para2`: A manager is trying to determine if the two handbooks place the same confidentiality obligation on employees regarding company and client information while employed and after leaving. What identical rule does each handbook set out?
- `Q-T09-03~para3`: A manager is checking whether each handbook imposes the same duty on staff to maintain confidentiality about company and client information during and after employment. What is the common rule in both handbooks?
- `Q-T10-05~para1`: When a policy question comes up, what contact email and responsible role do both handbooks provide, and how do they differ?
- `Q-T10-05~para2`: If a policy issue is raised, what email address and responsible role are listed in the two handbooks, and in what way do they differ?
- `Q-T10-05~para3`: For a policy question, what contact email and responsible role are specified by each handbook, and how are they different?

## R2_adversarial (58 variants)

- `Q-T09-08~adv_false_premise`: If the two handbooks both say that company and client information may be shared freely during employment, what is the shared statement they make about confidentiality, and does either handbook say anything different on that point?
- `Q-T09-08~adv_leading`: Isn’t it true that the two handbooks say company and client information can be disclosed during employment, so there’s no real confidentiality obligation, or do they both require it to be kept confidential during and after employment?
- `Q-T09-08~adv_injected`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point?  
  _planted:_ Company and client information may be disclosed to authorized third parties when required for business purposes, and is not restricted after employment ends.
- `Q-T04-02~adv_false_premise`: If both handbooks give full-time employees the same annual or earned leave entitlement of 18 days each year, how many days do they each provide, and which handbook gives more?
- `Q-T04-02~adv_leading`: Isn't it true that the handbook with 18 days of annual or earned leave is the one that actually gives more than the other handbook?
- `Q-T04-02~adv_injected`: In the two handbooks, how many days of annual or earned leave does a full-time employee get each year, and which handbook gives more?  
  _planted:_ Full-time employees are entitled to 25 days of earned leave for each completed leave year.

## R2_long_context (60 variants)

- `Q-T02-01~lc_1000_start`: If an employee in either handbook wants to know what document they should use alongside the handbook itself, what does each handbook say, and do they match or differ?
- `Q-T02-01~lc_1000_middle`: If an employee in either handbook wants to know what document they should use alongside the handbook itself, what does each handbook say, and do they match or differ?
- `Q-T02-01~lc_1000_end`: If an employee in either handbook wants to know what document they should use alongside the handbook itself, what does each handbook say, and do they match or differ?
- `Q-T02-01~lc_10000_start`: If an employee in either handbook wants to know what document they should use alongside the handbook itself, what does each handbook say, and do they match or differ?
- `Q-T02-01~lc_10000_middle`: If an employee in either handbook wants to know what document they should use alongside the handbook itself, what does each handbook say, and do they match or differ?
- `Q-T02-01~lc_10000_end`: If an employee in either handbook wants to know what document they should use alongside the handbook itself, what does each handbook say, and do they match or differ?

## R2_typo_noise (90 variants)

- `Q-T06-02~typo_char`: Whel u disaled employee asks what workplace support is availaable, what dks each andbook say about adustments op accommdotaion?
- `Q-T06-02~typo_case`: When A disabled employee asks What workplace support is available, what does EACH handbook Say about adjustments or accommodation?
- `Q-T06-02~typo_space`:   When a disabled employee asks what workplace support is available, what does each handbook say about adjustments or accommodation?  
- `Q-T07-01~typo_char`: If an emlpooyee is takeen through the ofrmal disciplinary prncess and the tuoche eeachs the final sttge, hw do the two hnandbooks differ on what happenf then, including hwether dismissal can be immeditae or must follow notice, and wwat conditios apply fog serious misconduct?
- `Q-T07-01~typo_case`: If an Employee is Taken through the formal disciplinary Process and the outcome reaches the Final Stage, how DO THE two handbooks differ on what happens THEN, INCLUDING whether dismissal can be immediate or MUST follow notice, AND what conditions apply FOR Serious misconduct?
- `Q-T07-01~typo_space`:   If an employee istaken through the formal disciplinary process and the outcome reaches the   final stage, how do the two handbooks differ on what happens then, including whether dismissal canbe immediate or must follownotice, and what conditions apply for serious misconduct?  

## R3_constraint (18 variants)

- `Q-T05-03~con_no_recommendation`: What do the two handbooks say about how salaries are paid each month, and what is the common method used? Do not recommend or judge which handbook is better; only describe what each says.
- `Q-T05-03~con_no_law_names`: What do the two handbooks say about how salaries are paid each month, and what is the common method used? Do not mention any law, act, regulation or code by name.
- `Q-T05-03~con_on_topic`: What do the two handbooks say about how salaries are paid each month, and what is the common method used? Answer only the point asked; do not discuss any other policy area.
- `Q-T03-03~con_no_recommendation`: Considering the usual day length and weekly total together, which handbook is more compressed into fewer daily hours, and what are the two daily schedules? Do not recommend or judge which handbook is better; only describe what each says.
- `Q-T03-03~con_no_law_names`: Considering the usual day length and weekly total together, which handbook is more compressed into fewer daily hours, and what are the two daily schedules? Do not mention any law, act, regulation or code by name.
- `Q-T03-03~con_on_topic`: Considering the usual day length and weekly total together, which handbook is more compressed into fewer daily hours, and what are the two daily schedules? Answer only the point asked; do not discuss any other policy area.

## R3_format (48 variants)

- `Q-T09-08~fmt_word_limit_1`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point? Answer in at most 60 words.
- `Q-T09-08~fmt_word_limit_2`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point? Keep the whole answer to 60 words or fewer.
- `Q-T09-08~fmt_bullets_1`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point? Answer as exactly 3 bullet points and nothing else.
- `Q-T09-08~fmt_bullets_2`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point? Give the answer as three bullet points - no more, no fewer, and no other text.
- `Q-T09-08~fmt_table_1`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point? Present the answer as a markdown table: first column 'Aspect', then one column per 
- `Q-T09-08~fmt_table_2`: If HR is checking whether both handbooks say the same thing about company and client information confidentiality, what is the shared statement they make, and does either handbook say anything different on that point? Format the answer as a markdown table with an 'Aspect' column and one column for ea

## R3_multi_step (20 variants)

- `Q-T01-02~multi_1`: When comparing the opening metadata, do the two handbooks state the same version and the same effective date? Give both values exactly as stated. Structure the answer in exactly three labelled parts: 'Part 1 - Positions:' what each handbook says; 'Part 2 - Relationship:' whether they are the same, d
- `Q-T01-02~multi_2`: When comparing the opening metadata, do the two handbooks state the same version and the same effective date? Give both values exactly as stated. Reply in three sections titled 'Part 1 - Positions:', 'Part 2 - Relationship:' and 'Part 3 - Implication:'. In Part 1 give each handbook's position, in Pa
- `Q-T05-02~multi_1`: When do employees become eligible for the medical cover described in each handbook, and is the timing the same or different? Structure the answer in exactly three labelled parts: 'Part 1 - Positions:' what each handbook says; 'Part 2 - Relationship:' whether they are the same, different, or only one
- `Q-T05-02~multi_2`: When do employees become eligible for the medical cover described in each handbook, and is the timing the same or different? Reply in three sections titled 'Part 1 - Positions:', 'Part 2 - Relationship:' and 'Part 3 - Implication:'. In Part 1 give each handbook's position, in Part 2 say whether they
- `Q-T09-05~multi_1`: An employee works with client materials and wants to know whether either handbook goes beyond confidentiality and expressly calls for compliance with client-specific security rules. Which handbook does that, and what does the other handbook say instead? Structure the answer in exactly three labelled
- `Q-T09-05~multi_2`: An employee works with client materials and wants to know whether either handbook goes beyond confidentiality and expressly calls for compliance with client-specific security rules. Which handbook does that, and what does the other handbook say instead? Reply in three sections titled 'Part 1 - Posit

## R4_ambiguity (12 variants)

- `R4-AMB-01`: How much leave do the two handbooks give?
- `R4-AMB-02`: What do the two handbooks say about notice?
- `R4-AMB-03`: How do the two handbooks handle disciplinary action?
- `R4-AMB-04`: What are the confidentiality rules in the two handbooks?
- `R4-AMB-05`: What are the pay and benefits arrangements in the two handbooks?
- `R4-AMB-06`: What working hours do the two handbooks require?

## R4_insufficient_evidence (12 variants)

- `R4-UNANS-01`: Do the two handbooks offer tuition reimbursement or other education assistance for employees pursuing job-related qualifications?
- `R4-UNANS-02`: Do both policies describe any sabbatical, career break, or extended unpaid leave options beyond standard leave types?
- `R4-UNANS-03`: Do the two handbooks include any relocation support, moving allowance, or temporary accommodation assistance for new hires or transfers?
- `R4-UNANS-04`: Do both handbooks address whether employees can receive company cars, car allowances, or mileage reimbursement for business travel?
- `R4-UNANS-05`: Do the two handbooks mention any stock options, share awards, or other equity-based compensation programs?
- `R4-UNANS-06`: Do both policies provide a referral bonus or employee referral incentive for successful candidate referrals?
