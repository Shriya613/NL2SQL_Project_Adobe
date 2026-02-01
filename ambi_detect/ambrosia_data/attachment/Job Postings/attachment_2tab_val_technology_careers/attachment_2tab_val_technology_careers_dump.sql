BEGIN TRANSACTION;
CREATE TABLE Analysts_Projects(
    analystId INTEGER,
    projectId INTEGER,
    FOREIGN KEY (analystId) REFERENCES Data_Analysts(id),
    FOREIGN KEY (projectId) REFERENCES Projects(id),
    PRIMARY KEY (analystId, projectId)
);
INSERT INTO "Analysts_Projects" VALUES(1,1);
INSERT INTO "Analysts_Projects" VALUES(2,2);
CREATE TABLE Companies(
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT,
    size TEXT
);
INSERT INTO "Companies" VALUES(1,'TechCorp','New York','Large');
INSERT INTO "Companies" VALUES(2,'Startup Inc.','San Francisco','Small');
CREATE TABLE Data_Analysts(
    id INTEGER PRIMARY KEY,
    name TEXT,
    experience INT,
    tools TEXT,
    grade TEXT 
);
INSERT INTO "Data_Analysts" VALUES(1,'Alice Johnson',4,'Excel, Tableau, R','Middle');
INSERT INTO "Data_Analysts" VALUES(2,'Bob Brown',7,'SQL, SAS, Python','Advanced');
CREATE TABLE Developers_Projects(
    developerId INTEGER,
    projectId INTEGER,
    FOREIGN KEY (developerId) REFERENCES Software_Developers(id),
    FOREIGN KEY (projectId) REFERENCES Projects(id),
    PRIMARY KEY (developerId, projectId)
);
INSERT INTO "Developers_Projects" VALUES(1,1);
INSERT INTO "Developers_Projects" VALUES(2,2);
CREATE TABLE Projects(
    id INTEGER PRIMARY KEY,
    description TEXT,
    startDate DATE,
    endDate DATE,
    companyId INTEGER,
    FOREIGN KEY (companyId) REFERENCES Companies(id)
);
INSERT INTO "Projects" VALUES(1,'Web Application Development','2021-08-01','2021-12-31',1);
INSERT INTO "Projects" VALUES(2,'Mobile App Design','2021-09-01','2021-12-31',2);
CREATE TABLE Software_Developers(
    id INTEGER PRIMARY KEY,
    name TEXT,
    experience INT,
    skills TEXT,
    grade TEXT 
);
INSERT INTO "Software_Developers" VALUES(1,'John Doe',3,'Python, Java, C++','Middle');
INSERT INTO "Software_Developers" VALUES(2,'Jane Smith',6,'JavaScript, PHP, Ruby','High');
COMMIT;
