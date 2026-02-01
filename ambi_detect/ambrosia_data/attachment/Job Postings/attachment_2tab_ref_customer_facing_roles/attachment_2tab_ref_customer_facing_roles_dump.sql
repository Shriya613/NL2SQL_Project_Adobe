BEGIN TRANSACTION;
CREATE TABLE CS_Jobs (
    job_posting_id INTEGER,
    customer_service_advisor_id INTEGER,
    FOREIGN KEY(job_posting_id) REFERENCES Job_Postings(id),
    FOREIGN KEY(customer_service_advisor_id) REFERENCES Customer_Service_Advisors(id));
INSERT INTO "CS_Jobs" VALUES(1,1);
INSERT INTO "CS_Jobs" VALUES(1,3);
CREATE TABLE Customer_Service_Advisors (
    id INTEGER PRIMARY KEY,
    advisor_name TEXT,
    language_id INTEGER,
    FOREIGN KEY(language_id) REFERENCES Languages(id));
INSERT INTO "Customer_Service_Advisors" VALUES(1,'Alice',1);
INSERT INTO "Customer_Service_Advisors" VALUES(2,'Bob',3);
INSERT INTO "Customer_Service_Advisors" VALUES(3,'Carol',NULL);
INSERT INTO "Customer_Service_Advisors" VALUES(4,'Dave',5);
INSERT INTO "Customer_Service_Advisors" VALUES(5,'Eve',2);
CREATE TABLE IT_Jobs (
    job_posting_id INTEGER,
    it_support_technician_id INTEGER,
    FOREIGN KEY(job_posting_id) REFERENCES Job_Postings(id),
    FOREIGN KEY(it_support_technician_id) REFERENCES IT_Support_Technicians(id));
INSERT INTO "IT_Jobs" VALUES(2,4);
INSERT INTO "IT_Jobs" VALUES(2,5);
INSERT INTO "IT_Jobs" VALUES(3,2);
CREATE TABLE IT_Support_Technicians (
    id INTEGER PRIMARY KEY,
    technician_name TEXT,
    language_id INTEGER,
    FOREIGN KEY(language_id) REFERENCES Languages(id));
INSERT INTO "IT_Support_Technicians" VALUES(1,'Frank',1);
INSERT INTO "IT_Support_Technicians" VALUES(2,'Grace',3);
INSERT INTO "IT_Support_Technicians" VALUES(3,'Hannah',NULL);
INSERT INTO "IT_Support_Technicians" VALUES(4,'Isaac',5);
INSERT INTO "IT_Support_Technicians" VALUES(5,'Jackie',2);
CREATE TABLE Job_Postings (
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    location TEXT,
    salary DECIMAL(10,2),
    posted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
INSERT INTO "Job_Postings" VALUES(1,'CS Advisor','Assist customers via phone and email.','New York',60000,'2024-03-13 20:10:28');
INSERT INTO "Job_Postings" VALUES(2,'IT Support Tech','Troubleshoot hardware and software issues.','San Francisco',70000,'2024-03-13 20:10:28');
INSERT INTO "Job_Postings" VALUES(3,'Manager','Oversee operations of a department.','Los Angeles',90000,'2024-03-13 20:10:28');
INSERT INTO "Job_Postings" VALUES(4,'Engineer','Design and develop new products.','Chicago',80000,'2024-03-13 20:10:28');
INSERT INTO "Job_Postings" VALUES(5,'Analyst','Analyze data and provide insights.','Miami',75000,'2024-03-13 20:10:28');
CREATE TABLE Languages (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE);
INSERT INTO "Languages" VALUES(1,'English');
INSERT INTO "Languages" VALUES(2,'Spanish');
INSERT INTO "Languages" VALUES(3,'French');
INSERT INTO "Languages" VALUES(4,'German');
INSERT INTO "Languages" VALUES(5,'Chinese');
COMMIT;
