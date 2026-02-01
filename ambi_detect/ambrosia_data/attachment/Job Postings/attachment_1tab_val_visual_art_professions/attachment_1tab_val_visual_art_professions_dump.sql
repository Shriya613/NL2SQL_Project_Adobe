BEGIN TRANSACTION;
CREATE TABLE AppliesToJobPosting(
    Apply_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Job_ID INTEGRER,
    Applicant_Name NVARCHAR(100),
    Resume_Path TEXT,
    Application_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Job_ID) REFERENCES JobPosting(ID));
INSERT INTO "AppliesToJobPosting" VALUES(1,1,'John Doe','/path/to/resume.pdf','2023-05-02 09:30:00');
INSERT INTO "AppliesToJobPosting" VALUES(2,2,'Jane Smith','/path/to/another_resume.docx','2023-05-02 09:35:00');
INSERT INTO "AppliesToJobPosting" VALUES(3,3,'Michael Johnson','/path/to/portfolio.html','2023-05-02 09:40:00');
INSERT INTO "AppliesToJobPosting" VALUES(4,4,'Emily Brown','/path/to/portfolio_video.mp4','2023-05-02 09:45:00');
INSERT INTO "AppliesToJobPosting" VALUES(5,5,'David Wilson','/path/to/event_photos.zip','2023-05-02 09:50:00');
CREATE TABLE Employer(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name NVARCHAR(100));
INSERT INTO "Employer" VALUES(1,'ABC Studio');
INSERT INTO "Employer" VALUES(2,'XYZ Agency');
INSERT INTO "Employer" VALUES(3,'PQR Production House');
INSERT INTO "Employer" VALUES(4,'LMN Advertising');
CREATE TABLE JobPosting(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title NVARCHAR(100),
    Description TEXT,
    Date DATETIME DEFAULT CURRENT_TIMESTAMP,
    Status );
INSERT INTO "JobPosting" VALUES(1,'Full Time Photographer','Looking for a full time photographer with experience in wedding photography.','2023-05-01 00:00:00','Open');
INSERT INTO "JobPosting" VALUES(2,'Part Time Videographer','We need a part time videographer who can create corporate videos.','2023-05-01 00:00:00','Filled');
INSERT INTO "JobPosting" VALUES(3,'Graphic Design Internship','Seeking an intern to assist our design team.','2023-05-01 00:00:00','Closed');
INSERT INTO "JobPosting" VALUES(4,'Senior Video Editor','Need a senior video editor for documentary filmmaking projects.','2023-05-01 00:00:00','Open');
INSERT INTO "JobPosting" VALUES(5,'Event Photographer','Hiring event photographers for various occasions.','2023-05-01 00:00:00','Open');
CREATE TABLE Visual_Art_Profession(
    Profession_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Profession NVARCHAR(100),
    Price REAL ,
    Experience_Level VARCHAR(20),
    Specialization VARCHAR(100));
INSERT INTO "Visual_Art_Profession" VALUES(1,'Photographer',500.0,'Experienced','Wedding Photography');
INSERT INTO "Visual_Art_Profession" VALUES(2,'Photographer',600.0,'Expert','Portrait Photography');
INSERT INTO "Visual_Art_Profession" VALUES(3,'Videographer',500.0,'Beginner','Corporate Videos');
INSERT INTO "Visual_Art_Profession" VALUES(4,'Videographer',700.0,'Intermediate','Documentary Filmmaking');
INSERT INTO "Visual_Art_Profession" VALUES(5,'Graphic Designer',800.0,'Advanced','Digital Art');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Visual_Art_Profession',5);
INSERT INTO "sqlite_sequence" VALUES('Employer',4);
INSERT INTO "sqlite_sequence" VALUES('JobPosting',5);
INSERT INTO "sqlite_sequence" VALUES('AppliesToJobPosting',5);
COMMIT;
