BEGIN TRANSACTION;
CREATE TABLE Applications(
    application_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_name TEXT,
    email TEXT,
    resume_path TEXT,
    status TEXT ,
    job_posting_id INTEGER,
    FOREIGN KEY(job_posting_id) REFERENCES JobPostings(posting_id)
);
INSERT INTO "Applications" VALUES(1,'John Doe','johndoe@email.com','/resume/john_doe.pdf','Pending',1);
INSERT INTO "Applications" VALUES(2,'Jane Smith','janesmith@email.com','/resume/jane_smith.pdf','Accepted',2);
INSERT INTO "Applications" VALUES(3,'Mike Brown','mikebrown@email.com','/resume/mike_brown.pdf','Rejected',3);
INSERT INTO "Applications" VALUES(4,'Emily Johnson','emilyjohnson@email.com','/resume/emily_johnson.pdf','Pending',4);
INSERT INTO "Applications" VALUES(5,'Tom Green','tomgreen@email.com','/resume/tom_green.pdf','Accepted',5);
CREATE TABLE Companies(
    company_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    location TEXT
);
INSERT INTO "Companies" VALUES(1,'Tech Company One','New York');
INSERT INTO "Companies" VALUES(2,'Software Inc.','San Francisco');
INSERT INTO "Companies" VALUES(3,'Web Solutions','Los Angeles');
INSERT INTO "Companies" VALUES(4,'App Developers','Chicago');
INSERT INTO "Companies" VALUES(5,'Technology Group','Washington DC');
CREATE TABLE Designer_Tools(
    designer_id INTEGER,
    tool_name TEXT,
    FOREIGN KEY(designer_id) REFERENCES Designers(id),
    FOREIGN KEY(tool_name) REFERENCES Tools(tool_name)
);
INSERT INTO "Designer_Tools" VALUES(1,'Adobe Creative Suite');
INSERT INTO "Designer_Tools" VALUES(1,'Illustrator');
INSERT INTO "Designer_Tools" VALUES(2,'Photoshop');
INSERT INTO "Designer_Tools" VALUES(3,'InDesign');
INSERT INTO "Designer_Tools" VALUES(4,'Premiere Pro');
INSERT INTO "Designer_Tools" VALUES(1,'Photoshop');
INSERT INTO "Designer_Tools" VALUES(3,'Photoshop');
INSERT INTO "Designer_Tools" VALUES(4,'Photoshop');
INSERT INTO "Designer_Tools" VALUES(5,'Photoshop');
CREATE TABLE Designers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INT
);
INSERT INTO "Designers" VALUES(1,'Alice',3);
INSERT INTO "Designers" VALUES(2,'Bob',2);
INSERT INTO "Designers" VALUES(3,'Carol',5);
INSERT INTO "Designers" VALUES(4,'Dave',1);
INSERT INTO "Designers" VALUES(5,'Eve',4);
CREATE TABLE JobPostings(
    posting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    salary INT,
    posted_date DATE,
    company_id INTEGER,
    FOREIGN KEY(company_id) REFERENCES Companies(company_id)
);
INSERT INTO "JobPostings" VALUES(1,'UX/UI Designer','We need a UX/UI designer to create user interfaces for our new product line',70000,'2022-08-01',1);
INSERT INTO "JobPostings" VALUES(2,'Graphic Designer','Our team needs an experienced graphic designer to work on various projects',60000,'2022-09-01',2);
INSERT INTO "JobPostings" VALUES(3,'Web Developer','We are looking for a web developer to join our growing team',75000,'2022-08-05',3);
INSERT INTO "JobPostings" VALUES(4,'Mobile App Developer','We want a mobile app developer to help us build our next big app',80000,'2022-08-10',4);
INSERT INTO "JobPostings" VALUES(5,'Data Analyst','Looking for data analyst to support business decisions',65000,'2022-08-15',5);
CREATE TABLE Tools(
    tool_name TEXT PRIMARY KEY,
    description TEXT
);
INSERT INTO "Tools" VALUES('Adobe Creative Suite','A collection of graphic design software');
INSERT INTO "Tools" VALUES('Illustrator','A vector graphics editor');
INSERT INTO "Tools" VALUES('Photoshop','An image editing software');
INSERT INTO "Tools" VALUES('InDesign','Page layout and typesetting software');
INSERT INTO "Tools" VALUES('Premiere Pro','Video editing software');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Designers',5);
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('JobPostings',5);
INSERT INTO "sqlite_sequence" VALUES('Applications',5);
COMMIT;
