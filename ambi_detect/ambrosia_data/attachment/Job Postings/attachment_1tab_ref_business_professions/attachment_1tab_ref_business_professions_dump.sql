BEGIN TRANSACTION;
CREATE TABLE Business_Profession(
    id INTEGER PRIMARY KEY,
    profession TEXT);
INSERT INTO "Business_Profession" VALUES(1,'Sales Manager');
INSERT INTO "Business_Profession" VALUES(2,'Sales Associate');
INSERT INTO "Business_Profession" VALUES(3,'Marketing Specialist');
INSERT INTO "Business_Profession" VALUES(4,'Product Manager');
INSERT INTO "Business_Profession" VALUES(5,'Data Analyst');
CREATE TABLE Company(
    id INTEGER PRIMARY KEY,
    name TEXT,
    industry TEXT,
    location TEXT);
INSERT INTO "Company" VALUES(1,'TechCorp','Technology','New York');
INSERT INTO "Company" VALUES(2,'Finance Inc.','Financial Services','Los Angeles');
INSERT INTO "Company" VALUES(3,'Health Solutions','Healthcare','Chicago');
INSERT INTO "Company" VALUES(4,'EcoFriendly','Environmental Services','San Francisco');
INSERT INTO "Company" VALUES(5,'Travel World','Travel & Tourism','Miami');
CREATE TABLE Job_Posting(
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    required_skills TEXT,
    salary REAL,
    post_date DATE,
    business_profession_id INTEGER,
    working_arrangement_id INTEGER,
    company_id INTEGER,
    FOREIGN KEY (business_profession_id) REFERENCES Business_Profession(id),
    FOREIGN KEY (working_arrangement_id) REFERENCES Working_Arrangement(id),
    FOREIGN KEY (company_id) REFERENCES Company(id));
INSERT INTO "Job_Posting" VALUES(1,'Senior Sales Manager','Lead a team of sales representatives...','Strong communication skills, leadership experience...',80000.0,'2022-06-01',1,1,1);
INSERT INTO "Job_Posting" VALUES(2,'Regional Sales Manager','Manage multiple territories across the country...','Strong negotiation skills, knowledge of CRM systems...',70000.0,'2022-06-02',1,2,2);
INSERT INTO "Job_Posting" VALUES(3,'Digital Marketing Specialist','Develop and execute digital marketing strategies...','SEO, SEM, social media management...',60000.0,'2022-06-01',3,1,1);
INSERT INTO "Job_Posting" VALUES(4,'Market Research Specialist','Analyze market trends, customer behavior...','Statistical analysis, data visualization...',55000.0,'2022-06-02',3,3,3);
INSERT INTO "Job_Posting" VALUES(5,'Software Engineer','Design and develop software applications...','JavaScript, Python, HTML/CSS...',90000.0,'2022-06-01',4,4,4);
CREATE TABLE Working_Arrangement(
    id INTEGER PRIMARY KEY,
    arrangement TEXT);
INSERT INTO "Working_Arrangement" VALUES(1,'Hybrid');
INSERT INTO "Working_Arrangement" VALUES(2,'Remote');
INSERT INTO "Working_Arrangement" VALUES(3,'Onsite');
INSERT INTO "Working_Arrangement" VALUES(4,'Flexible');
INSERT INTO "Working_Arrangement" VALUES(5,'In-office');
COMMIT;
