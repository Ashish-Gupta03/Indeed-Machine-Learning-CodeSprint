Tagging Raw Job Descriptions
This challenge asks for a machine learning solution to accurately assign tags given the information in the job descriptions. We are interested in only these twelve tags:
part-time-job
full-time-job
hourly-wage
salary
associate-needed
bs-degree-needed
ms-or-phd-needed
licence-needed
1-year-experience-needed
2-4-years-experience-needed
5-plus-years-experience-needed
supervising-job

Note that some tags are mutually exclusive, meaning, for example, that a job can not require both - years of experience and  5 years of experience. It is also possible that the job description does not contain any information relevant for tagging.

The output file, pred.tsv (max allowed size is 10MB). The file should contain a space-separated list of tags -- the order of the tags does not matter -- for each of the job descriptions from the file test.tsv in that same order. You can choose not to tag any description by providing an empty list of tags.