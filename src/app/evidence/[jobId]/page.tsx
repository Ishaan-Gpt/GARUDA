import JobResultModule from "@/modules/evidence/JobResultModule";

export default async function EvidenceJobResultPage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = await params;
  return <JobResultModule jobId={jobId} />;
}
