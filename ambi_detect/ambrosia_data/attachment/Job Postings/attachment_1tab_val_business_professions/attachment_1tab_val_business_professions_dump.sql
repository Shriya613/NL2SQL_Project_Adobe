BEGIN TRANSACTION;
CREATE TABLE Applicants(
    ApplicantID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber VARCHAR(15)
);
INSERT INTO "Applicants" VALUES(1,'John','Doe','john.doe@example.com','+1234567890');
INSERT INTO "Applicants" VALUES(2,'Jane','Smith','jane.smith@example.com','+0987654321');
INSERT INTO "Applicants" VALUES(3,'Michael','Johnson','michael.johnson@example.com','+1357924680');
INSERT INTO "Applicants" VALUES(4,'Emily','Williams','emily.williams@example.com','+0123456789');
INSERT INTO "Applicants" VALUES(5,'David','Brown','david.brown@example.com','+9876543210');
CREATE TABLE Companies(
    CompanyID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Industry TEXT
);
INSERT INTO "Companies" VALUES(1,'Tech Solutions Inc.','Technology');
INSERT INTO "Companies" VALUES(2,'Healthy Living LLC','Healthcare');
INSERT INTO "Companies" VALUES(3,'Financial Services Co.','Banking');
INSERT INTO "Companies" VALUES(4,'Green Energy Corp.','Energy');
INSERT INTO "Companies" VALUES(5,'Creative Studios Ltd.','Entertainment');
CREATE TABLE JobPostings(
    JobID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Description TEXT,
    CompanyID INTEGER,
    Business_Profession TEXT, -- This could contain multiple professions separated by commas or spaces
    Working_Arrangement TEXT, -- Could contain values like Hybrid, Remote, In-Office, etc.
    LocationID INTEGER,
    PostedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    CloseDate DATETIME,
    FOREIGN KEY (CompanyID) REFERENCES Companies(CompanyID),
    FOREIGN KEY (LocationID) REFERENCES Locations(LocationID)
);
INSERT INTO "JobPostings" VALUES(1,'Sales Manager','Manage sales team and drive revenue growth.',1,'Sales Manager','Hybrid',1,'CURRENT_TIMESTAMP','2023-01-31');
INSERT INTO "JobPostings" VALUES(2,'Senior Sales Manager','Lead and manage a high-performing sales team.',2,'Sales Manager','In-Office',2,'CURRENT_TIMESTAMP','2023-01-31');
INSERT INTO "JobPostings" VALUES(3,'Marketing Specialist','Create and execute marketing strategies to increase brand awareness.',1,'Marketing Specialist','Hybrid',1,'CURRENT_TIMESTAMP','2023-01-31');
INSERT INTO "JobPostings" VALUES(4,'Junior Marketing Specialist','Develop and implement effective marketing campaigns.',3,'Marketing Specialist','Remote',3,'CURRENT_TIMESTAMP','2023-01-31');
CREATE TABLE Locations(
    LocationID INTEGER PRIMARY KEY AUTOINCREMENT,
    City TEXT,
    State TEXT,
    Country TEXT
);
INSERT INTO "Locations" VALUES(1,'New York','NY','USA');
INSERT INTO "Locations" VALUES(2,'Los Angeles','CA','USA');
INSERT INTO "Locations" VALUES(3,'Chicago','IL','USA');
INSERT INTO "Locations" VALUES(4,'Toronto','ON','Canada');
INSERT INTO "Locations" VALUES(5,'London','England','UK');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Locations',5);
INSERT INTO "sqlite_sequence" VALUES('Applicants',5);
INSERT INTO "sqlite_sequence" VALUES('JobPostings',4);
COMMIT;
