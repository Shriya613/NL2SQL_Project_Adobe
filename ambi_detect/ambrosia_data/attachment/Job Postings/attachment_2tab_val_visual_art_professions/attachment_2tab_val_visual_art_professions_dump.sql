BEGIN TRANSACTION;
CREATE TABLE Applications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicantId INTEGER,
    applicationDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    coverLetter TEXT,
    status TEXT );
INSERT INTO "Applications" VALUES(1,1,'2022-01-01 10:00:00','Dear Hiring Manager...','Pending');
INSERT INTO "Applications" VALUES(2,2,'2022-01-02 10:00:00','Thank you for considering my application...','Accepted');
INSERT INTO "Applications" VALUES(3,3,'2022-01-03 10:00:00','I have extensive experience working on projects similar to this one...','Rejected');
CREATE TABLE JobPosts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jobTitle TEXT,
    description TEXT,
    requiredSkills TEXT,
    budget NUMERIC );
INSERT INTO "JobPosts" VALUES(1,'Event Photographer','Capture photos at a corporate event.','Adobe Lightroom, Camera Operation',2000);
INSERT INTO "JobPosts" VALUES(2,'Freelance Videographer','Shoot and edit promotional video for small business.','Final Cut Pro, Cinematography',3000);
INSERT INTO "JobPosts" VALUES(3,'Product Photographer','Take high quality images of products for online store.','Studio Lighting, Retouching',1500);
CREATE TABLE Photographers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photographerName TEXT,
    price NUMERIC ,
    experience INT,
    specialization TEXT);
INSERT INTO "Photographers" VALUES(1,'John Smith',500,10,'Wedding');
INSERT INTO "Photographers" VALUES(2,'Jane Doe',600,8,'Portraiture');
INSERT INTO "Photographers" VALUES(3,'Mike Johnson',700,5,'Landscape');
CREATE TABLE SkillRequirements(
    requirementID INTEGER PRIMARY KEY AUTOINCREMENT,
    jobPostId INTEGER,
    skillId INTEGER,
    FOREIGN KEY (jobPostId) REFERENCES JobPosts(id),
    FOREIGN KEY (skillId) REFERENCES Skills(skillId));
INSERT INTO "SkillRequirements" VALUES(1,1,1);
INSERT INTO "SkillRequirements" VALUES(2,1,2);
INSERT INTO "SkillRequirements" VALUES(3,2,3);
INSERT INTO "SkillRequirements" VALUES(4,2,4);
INSERT INTO "SkillRequirements" VALUES(5,3,5);
CREATE TABLE Skills(
    skillId INTEGER PRIMARY KEY AUTOINCREMENT,
    skillName TEXT UNIQUE);
INSERT INTO "Skills" VALUES(1,'Adobe Lightroom');
INSERT INTO "Skills" VALUES(2,'Camera Operation');
INSERT INTO "Skills" VALUES(3,'Final Cut Pro');
INSERT INTO "Skills" VALUES(4,'Cinematography');
INSERT INTO "Skills" VALUES(5,'Studio Lighting');
INSERT INTO "Skills" VALUES(6,'Retouching');
CREATE TABLE Videographers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    videographerName TEXT,
    price NUMERIC ,
    experience INT,
    specialization TEXT);
INSERT INTO "Videographers" VALUES(1,'David Wilson',500,7,'Corporate Events');
INSERT INTO "Videographers" VALUES(2,'Emily Brown',600,9,'Documentary');
INSERT INTO "Videographers" VALUES(3,'William Taylor',700,3,'Music Video');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Photographers',3);
INSERT INTO "sqlite_sequence" VALUES('Videographers',3);
INSERT INTO "sqlite_sequence" VALUES('JobPosts',3);
INSERT INTO "sqlite_sequence" VALUES('Applications',3);
INSERT INTO "sqlite_sequence" VALUES('Skills',6);
INSERT INTO "sqlite_sequence" VALUES('SkillRequirements',5);
COMMIT;
