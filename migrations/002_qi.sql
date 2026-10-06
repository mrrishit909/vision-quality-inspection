-- Domain model. Names follow the blueprint's schema; additions are marked (+).

CREATE TABLE factory (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, site text NOT NULL);
SELECT enable_tenant_rls('factory');

CREATE TABLE product (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, code text NOT NULL, name text NOT NULL, family text NOT NULL);
SELECT enable_tenant_rls('product');

CREATE TABLE line (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, factory_id uuid NOT NULL REFERENCES factory, product_id uuid NOT NULL REFERENCES product, name text NOT NULL, seed int NOT NULL, frames_seen int NOT NULL DEFAULT 0);   -- (+) seed of the simulated line, frame counter
SELECT enable_tenant_rls('line');

CREATE TABLE camera (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, line_id uuid NOT NULL REFERENCES line, name text NOT NULL, fps int NOT NULL, width int NOT NULL, height int NOT NULL);
SELECT enable_tenant_rls('camera');

-- Frames. Pixels are kept (PNG) for the training set, every reject and every audited pass; other passes keep only their hash.
CREATE TABLE image (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, camera_id uuid NOT NULL REFERENCES camera, frame_no int NOT NULL, uri text NOT NULL, sha256 text NOT NULL, captured_at timestamptz NOT NULL DEFAULT now(), png bytea, sim_truth jsonb NOT NULL);   -- (+) png, sim_truth: the generator's label, used only to score
SELECT enable_tenant_rls('image');
CREATE INDEX image_frame ON image (tenant_id, camera_id, frame_no);

CREATE TABLE model_version (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, line_id uuid NOT NULL REFERENCES line, version int NOT NULL, artifact bytea NOT NULL, sha256 text NOT NULL, metrics jsonb NOT NULL, training jsonb NOT NULL, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (line_id, version));
SELECT enable_tenant_rls('model_version');

CREATE TABLE deployment (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, line_id uuid NOT NULL REFERENCES line, model_version_id uuid NOT NULL REFERENCES model_version, mode text NOT NULL CHECK (mode IN ('full', 'canary')), share numeric NOT NULL CHECK (share > 0 AND share <= 1), status text NOT NULL CHECK (status IN ('active', 'promoted', 'rolled_back', 'retired')), created_by uuid NOT NULL, decided_by uuid, decision jsonb, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('deployment');
CREATE UNIQUE INDEX one_active_per_mode ON deployment (line_id, mode) WHERE status = 'active';

CREATE TABLE inspection_run (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, line_id uuid NOT NULL REFERENCES line, frames int NOT NULL, fps int NOT NULL, params jsonb NOT NULL, summary jsonb NOT NULL DEFAULT '{}', created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now());   -- (+) one stretch of the line's stream
SELECT enable_tenant_rls('inspection_run');

CREATE TABLE inspection (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, image_id uuid NOT NULL REFERENCES image, run_id uuid NOT NULL REFERENCES inspection_run, model_version_id uuid NOT NULL REFERENCES model_version, deployment_id uuid NOT NULL REFERENCES deployment, arm text NOT NULL, score real NOT NULL, decision text NOT NULL CHECK (decision IN ('pass', 'reject')), latency_ms real NOT NULL, audited boolean NOT NULL DEFAULT false, audit_found_defect boolean, anomaly_map real[], inspected_at timestamptz NOT NULL DEFAULT now());   -- (+) arm, audit of passed parts, map for rejects
SELECT enable_tenant_rls('inspection');
CREATE INDEX inspection_run_idx ON inspection (tenant_id, run_id, decision);

CREATE TABLE defect (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, inspection_id uuid NOT NULL REFERENCES inspection, defect_type text NOT NULL, bbox int[] NOT NULL, confidence real);
SELECT enable_tenant_rls('defect');

CREATE TABLE annotation (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, image_id uuid NOT NULL REFERENCES image, label text NOT NULL CHECK (label IN ('good', 'defect')), defect_type text, split text NOT NULL CHECK (split IN ('train', 'calibration', 'feedback')), source text NOT NULL, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now());   -- (+) split, source
SELECT enable_tenant_rls('annotation');

CREATE TABLE operator_feedback (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, inspection_id uuid NOT NULL REFERENCES inspection UNIQUE, verdict text NOT NULL CHECK (verdict IN ('false_positive', 'confirmed_defect')), defect_type text, operator text NOT NULL, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('operator_feedback');
