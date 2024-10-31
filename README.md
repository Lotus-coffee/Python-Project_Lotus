# Python-Project_Lotus
This goal of this project is to predict whether an applicant is approval for a loan.

I  downloaded the data on kaggle. In this project, I will use area under the ROC curve to evaluate the results.

For the submission, there will be two columns with just ID and loan status, which can 
tell if a applicant is approval for a loan.

Conclusion: 89.34% of the applicants can get loan approval from the bank. the prediction accuracy is 93.72%.

About the dataset:

The data includes a test data file and a train data file, with columns as applicant id, applicant's age, applicant's income, applicant's home ownership, applicant's employment length, loan intention, loan grade,loan amount, loan interest rate, loan percent income, and loan approval.

1. person_age: The age of the borrower when securing the loan.
2. person_income: The borrower’s annual earnings at the time of the loan.
3. person_home_ownership: Type of home ownership.
   rent: The borrower is currently renting a property.
   mortgage: The borrower has a mortgage on the property they own.
   own: The borrower owns their home outright.
   other: Other categories of home ownership that may be specific to the dataset.
4. person_emp_length: The amount of time in years that borrower is employed.
5. loan_intent: Loan purpose.
6. loan_grade: Classification system based on credit history, collateral quality, and likelihood of repayment of the principal and interest.
   A: The borrower has a high creditworthiness, indicating low risk.
   B: The borrower is relatively low-risk, but not as creditworthy as Grade A.
   C: The borrower’s creditworthiness is moderate.
   D: The borrower is considered to have higher risk compared to previous grades.
   E: The borrower’s creditworthiness is lower, indicating a higher risk.
   F: The borrower poses a significant credit risk.
   G: The borrower’s creditworthiness is the lowest, signifying the highest risk.
7. loan_amnt: Total amount of the loan.
8. loan_int_rate: Interest rate of the loan.
9. loan_status: Dummy variable indicating default (1) or non-default (0).
   0: Non-default - The borrower successfully repaid the loan as agreed, and there was no default.
   1: Default - The borrower failed to repay the loan according to the agreed-upon terms and defaulted.
10. loan_percent_income: Ratio between the loan amount and the annual income.
11. cb_person_cred_hist_length: The number of years of personal history since the first loan taken from borrower.
12. cb_person_default_on_file: Indicate if the person has previously defaulted.



