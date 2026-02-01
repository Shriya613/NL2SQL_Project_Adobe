BEGIN TRANSACTION;
CREATE TABLE Applications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicantName TEXT,
    applicationDate DATETIME DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE Companies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Companies" VALUES(1,'Tech Giants');
INSERT INTO "Companies" VALUES(2,'Startups Inc.');
INSERT INTO "Companies" VALUES(3,'Software Solutions Ltd.');
INSERT INTO "Companies" VALUES(4,'Technology Firm');
INSERT INTO "Companies" VALUES(5,'Data Analytics Co.');
CREATE TABLE CompanyLocations(
    companyId INTEGER REFERENCES Companies(id) ON DELETE CASCADE,
    locationId INTEGER REFERENCES Locations(id) ON DELETE CASCADE,
    PRIMARY KEY (companyId, locationId));
INSERT INTO "CompanyLocations" VALUES(1,1);
CREATE TABLE InternshipApplications(
    internshipId INTEGER REFERENCES Internships(id) ON DELETE CASCADE,
    applicationId INTEGER REFERENCES Applications(id) ON DELETE CASCADE,
    PRIMARY KEY (internshipId, applicationId));
INSERT INTO "InternshipApplications" VALUES(1,NULL);
CREATE TABLE Internships(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT);
INSERT INTO "Internships" VALUES(1,'Web Development Intern','Develop web applications using Java and JavaScript');
INSERT INTO "Internships" VALUES(2,'Mobile Apps Intern','Design and develop mobile apps for Android and iOS platforms');
INSERT INTO "Internships" VALUES(3,'Data Science Intern','Analyze data sets and create predictive models');
INSERT INTO "Internships" VALUES(4,'Marketing Intern','Assist in marketing campaigns and social media management');
INSERT INTO "Internships" VALUES(5,'UI/UX Designer Intern','Create user interfaces and improve user experience');
CREATE TABLE Internships_Responsibilities(
    internshipId INTEGER REFERENCES Internships(id) ON DELETE CASCADE,
    responsibilityId INTEGER REFERENCES Responsibilities(id) ON DELETE CASCADE,
    PRIMARY KEY (internshipId, responsibilityId));
INSERT INTO "Internships_Responsibilities" VALUES(1,1);
INSERT INTO "Internships_Responsibilities" VALUES(1,2);
INSERT INTO "Internships_Responsibilities" VALUES(2,1);
INSERT INTO "Internships_Responsibilities" VALUES(3,1);
INSERT INTO "Internships_Responsibilities" VALUES(4,1);
INSERT INTO "Internships_Responsibilities" VALUES(5,1);
CREATE TABLE Locations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    state TEXT,
    country TEXT);
INSERT INTO "Locations" VALUES(1,'New York','NY','USA');
INSERT INTO "Locations" VALUES(2,'San Francisco','CA','USA');
INSERT INTO "Locations" VALUES(3,'London','England','UK');
INSERT INTO "Locations" VALUES(4,'Berlin','Germany','Germany');
INSERT INTO "Locations" VALUES(5,'Tokyo','Japan','Japan');
CREATE TABLE Responsibilities(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    responsibility TEXT);
INSERT INTO "Responsibilities" VALUES(1,'Research');
INSERT INTO "Responsibilities" VALUES(2,'Project Management');
INSERT INTO "Responsibilities" VALUES(3,'Quality Assurance');
INSERT INTO "Responsibilities" VALUES(4,'Customer Support');
INSERT INTO "Responsibilities" VALUES(5,'Content Writing');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Companies',5);
INSERT INTO "sqlite_sequence" VALUES('Locations',5);
INSERT INTO "sqlite_sequence" VALUES('Internships',5);
INSERT INTO "sqlite_sequence" VALUES('Responsibilities',5);
COMMIT;
