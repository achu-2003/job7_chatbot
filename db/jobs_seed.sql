-- Sample recruiting data for local testing (tenant 'default').
-- Idempotent via ON CONFLICT on the natural keys. Run with:
--   python scripts/init_jobs_db.py --seed

INSERT INTO departments (tenant_id, name) VALUES
    ('default', 'Engineering'),
    ('default', 'Product'),
    ('default', 'Sales')
ON CONFLICT (tenant_id, name) DO NOTHING;

INSERT INTO jobs (
    tenant_id, job_ref, title, department_id, location, employment_type,
    seniority, salary_min, salary_max, salary_currency, skills, description, status
)
SELECT
    'default', v.job_ref, v.title,
    (SELECT id FROM departments d WHERE d.tenant_id = 'default' AND d.name = v.dept),
    v.location, v.employment_type, v.seniority,
    v.salary_min, v.salary_max, 'INR', v.skills, v.description, 'OPEN'
FROM (VALUES
    ('JOB-AB1001', 'Senior Backend Engineer', 'Engineering', 'Bengaluru (Hybrid)',
     'full_time', 'senior', 2500000::numeric, 4000000::numeric,
     ARRAY['python','fastapi','postgresql','aws'],
     'Build and scale our API platform. Strong async Python + Postgres experience required.'),
    ('JOB-AB1002', 'Product Manager', 'Product', 'Remote (India)',
     'full_time', 'mid', 1800000::numeric, 2800000::numeric,
     ARRAY['product','roadmap','analytics'],
     'Own the discovery-to-delivery lifecycle for our core product area.'),
    ('JOB-AB1003', 'Enterprise Account Executive', 'Sales', 'Mumbai',
     'full_time', 'senior', 1500000::numeric, 3000000::numeric,
     ARRAY['saas sales','negotiation','crm'],
     'Close enterprise deals and grow strategic accounts across West India.')
) AS v(job_ref, title, dept, location, employment_type, seniority,
       salary_min, salary_max, skills, description)
ON CONFLICT (tenant_id, job_ref) DO NOTHING;
