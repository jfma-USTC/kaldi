// hmm/hmm-topology-2D.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "hmm-topology-2D.h"
#include "util/text-utils.h"


namespace kaldi {

	void HmmTopology_2D::GetPhoneToNumPdfClasses(std::vector<int32> *phone2num_pdf_classes) const {
		KALDI_ASSERT(!phones_.empty());
		phone2num_pdf_classes->clear();
		phone2num_pdf_classes->resize(phones_.back() + 1, -1);
		// resize() don't change old values but push_back new values as assigned
		for (size_t i = 0; i < phones_.size(); i++)
			(*phone2num_pdf_classes)[phones_[i]] = NumPdfClasses(phones_[i]); 
		    // make sure the size of phone2num_pdf_classes equal to the max number of phones_,
		    // so we can use phone2num_pdf_classes[phone_id] to get corresponding num_pdf_classes.
	}

	void HmmTopology_2D::Read(std::istream &is, bool binary) {
		ExpectToken(is, binary, "<Topology_2D>");
		if (!binary) {  // Text-mode read, different "human-readable" format.
			phones_.clear();
			phone2idx_.clear();
			entries_.clear();
			std::string token;
			// ��ʱ�ļ�ָ��λ�ڵڶ���
			while (!(is >> token).fail()) { // ���ļ�����ȡ�ַ���ʱ�� >> ���ո񡢻��н���
				if (token == "</Topology_2D>") { break; } // finished parsing.
				else  if (token != "<TopologyEntry_2D>") { // ����ڶ��в���<TopologyEntry_2D>�ͱ���
					KALDI_ERR << "Reading HmmTopology_2D object, expected </Topology_2D> or <TopologyEntry_2D>, got " << token;
				}
				else {
					ExpectToken(is, binary, "<ForPhones>");
					std::vector<int32> phones;
					std::string s;
					// ��һ��while�����ڶ�ȡ<ForPhones>�е���Ϣ������Щ���أ�
					while (1) { 
						is >> s; // ���ļ�����ȡ�ַ���ʱ�� >> ���ո񡢻��н���
						if (is.fail()) KALDI_ERR << "Reading HmmTopology_2D object, unexpected end of file while expecting phones.";
						if (s == "</ForPhones>") break;
						else {
							int32 phone;
							if (!ConvertStringToInteger(s, &phone))
								KALDI_ERR << "Reading HmmTopology_2D object, expected "
								<< "integer, got instead " << s;
							phones.push_back(phone);
						}
					}
					// �˴�isָ��Ӧ����</ForPhone>����һ��
					std::vector<HmmState_2D> this_entry; // һ��vector<HmmState_2D>��Ӧ��һ��phone
					std::string token;
					ReadToken(is, binary, &token); // ����token�ִ��д����'<TopDown>'��'<LeftRight>'
					// ��һ��while�����ڶ�ȡ<TopologyEntry_2D>�е���Ϣ������Щ���أ�
					while (token != "</TopologyEntry_2D>") {
						//if (token != "<TopDown>" || token != "<LeftRight>")
							//KALDI_ERR << "Expected <TopDown> or <LeftRight>, got instead " << token;
						if (token == "<TopDown>") {
							ReadToken(is, binary, &token); // ����token�ִ��д����'<State>'
							if (token != "<State>")
								KALDI_ERR << "Expected <State>, got instead " << token;
							if (!this_entry.size()) { 
							// ���this_entry��û����ʼ����������������<TopDown>...</TopDown>ת�ƣ���δ����<LeftRight>...</LeftRight>��
								while (token != "</TopDown>") {
									int32 state;
									ReadBasicType(is, binary, &state);
									if (state != static_cast<int32>(this_entry.size()))
										KALDI_ERR << "States are expected to be in order from zero, expected "
										<< this_entry.size() << ", got " << state;
									ReadToken(is, binary, &token);
									
									int32 forward_pdf_class = kNoPdf;  // -1 by default, means no pdf.
									if (token == "<PdfClass>") {
										ReadBasicType(is, binary, &forward_pdf_class);
										this_entry.push_back(HmmState_2D(forward_pdf_class));
										ReadToken(is, binary, &token);
										if (token == "<SelfLoopPdfClass>")
											KALDI_ERR << "pdf classes should be defined using <PdfClass> "
											<< "or <ForwardPdfClass>/<SelfLoopPdfClass> pair";
									}
									else if (token == "<ForwardPdfClass>") {
										int32 self_loop_pdf_class = kNoPdf;
										ReadBasicType(is, binary, &forward_pdf_class);
										ReadToken(is, binary, &token);
										if (token != "<SelfLoopPdfClass>")
											KALDI_ERR << "Expected <SelfLoopPdfClass>, got instead " << token;
										ReadBasicType(is, binary, &self_loop_pdf_class);
										this_entry.push_back(HmmState_2D(forward_pdf_class, self_loop_pdf_class));
										ReadToken(is, binary, &token);
									}
									else
										this_entry.push_back(HmmState_2D(forward_pdf_class));

									while (token == "<Transition>") {
										int32 dst_state;
										BaseFloat trans_prob;
										ReadBasicType(is, binary, &dst_state);
										ReadBasicType(is, binary, &trans_prob);
										this_entry.back().trans_top_down.push_back(std::make_pair(dst_state, trans_prob));
										ReadToken(is, binary, &token);
									}
									if (token != "</State>")
										KALDI_ERR << "Expected </State>, got instead " << token;
									ReadToken(is, binary, &token); // token should be <State> or </TopDown> here
								}
							}else {
							// ���this_entry�Ѿ�����ʼ��һ�Σ����Ѿ�������<LeftRight>...</LeftRight>��ֻ��Ҫ���trans_top_down���Լ��ɣ�
								while (token != "</TopDown>") {
									int32 state;
									ReadBasicType(is, binary, &state);
									if (state >= this_entry.size()) {
										KALDI_ERR << "Phone's pdf number for <TopDown> and <LeftRight> don't match, <LeftRight> has " << this_entry.size() << " pdfs";
									}
									ReadToken(is, binary, &token);
									int32 forward_pdf_class = kNoPdf;  // -1 by default, means no pdf.

									if (token == "<PdfClass>") {
										ReadBasicType(is, binary, &forward_pdf_class);
										if (this_entry.at(state).self_loop_pdf_class != forward_pdf_class) 
											KALDI_ERR << "Phone's PdfClass don't match, LeftRight's state " << state << " has PdfClass for "
												      << this_entry.at(state).self_pdf_class << " but Topdown's state " << state << " has PdfClass for "
												      << forward_pdf_class;
										ReadToken(is, binary, &token);
										if (token == "<SelfLoopPdfClass>")
											KALDI_ERR << "pdf classes should be defined using <PdfClass> "
											          << "or <ForwardPdfClass>/<SelfLoopPdfClass> pair";
									}else if (token == "<ForwardPdfClass>") {
										int32 self_loop_pdf_class = kNoPdf;
										ReadBasicType(is, binary, &forward_pdf_class);
										ReadToken(is, binary, &token);
										if (token != "<SelfLoopPdfClass>")
											KALDI_ERR << "Expected <SelfLoopPdfClass>, got instead " << token;
										ReadBasicType(is, binary, &self_loop_pdf_class);
										if (this_entry.at(state).self_loop_pdf_class != self_loop_pdf_class ||
											this_entry.at(state).forward_pdf_class != forward_pdf_class) 
											KALDI_ERR << "Phone's PdfClass don't match, LeftRight's state " << state << " has self_loop_pdf_class for "
												      << this_entry.at(state).self_loop_pdf_class << "and forward_pdf_class for "
												      << this_entry.at(state).forward_pdf_class << " but Topdown's state " << state   
												      << " has self_loop_pdf_class for " << self_loop_pdf_class << "and forward_pdf_class for "
												      << forward_pdf_class;
										ReadToken(is, binary, &token);
									}else {
										if(this_entry.at(state).forward_pdf_class != forward_pdf_class)
											KALDI_ERR << "Phone's PdfClass don't match, LeftRight's state " << state << " has PdfClass for "
											          << this_entry.at(state).self_pdf_class << " but Topdown's state " << state << " has PdfClass for "
											          << forward_pdf_class;
									}

									while (token == "<Transition>") {
										int32 dst_state;
										BaseFloat trans_prob;
										ReadBasicType(is, binary, &dst_state);
										ReadBasicType(is, binary, &trans_prob);
										this_entry.at(state).trans_top_down.push_back(std::make_pair(dst_state, trans_prob));
										ReadToken(is, binary, &token);
									}
									if (token != "</State>")
										KALDI_ERR << "Expected </State>, got instead " << token;
									ReadToken(is, binary, &token);
								}
							}
						}

						if (token == "<LeftRight>") {
							ReadToken(is, binary, &token); // ����token�ִ��д����'<State>'
							if (token != "<State>")
								KALDI_ERR << "Expected <State>, got instead " << token;
							if (!this_entry.size()) {
								// ���this_entry��û����ʼ����������������<LeftRight>...</LeftRight>ת�ƣ���δ����<TopDown>...</TopDown>��
								while (token != "</LeftRight>") {
									int32 state;
									ReadBasicType(is, binary, &state);
									if (state != static_cast<int32>(this_entry.size()))
										KALDI_ERR << "States are expected to be in order from zero, expected "
										<< this_entry.size() << ", got " << state;
									ReadToken(is, binary, &token);

									int32 forward_pdf_class = kNoPdf;  // -1 by default, means no pdf.
									if (token == "<PdfClass>") {
										ReadBasicType(is, binary, &forward_pdf_class);
										this_entry.push_back(HmmState_2D(forward_pdf_class));
										ReadToken(is, binary, &token);
										if (token == "<SelfLoopPdfClass>")
											KALDI_ERR << "pdf classes should be defined using <PdfClass> "
											<< "or <ForwardPdfClass>/<SelfLoopPdfClass> pair";
									}
									else if (token == "<ForwardPdfClass>") {
										int32 self_loop_pdf_class = kNoPdf;
										ReadBasicType(is, binary, &forward_pdf_class);
										ReadToken(is, binary, &token);
										if (token != "<SelfLoopPdfClass>")
											KALDI_ERR << "Expected <SelfLoopPdfClass>, got instead " << token;
										ReadBasicType(is, binary, &self_loop_pdf_class);
										this_entry.push_back(HmmState_2D(forward_pdf_class, self_loop_pdf_class));
										ReadToken(is, binary, &token);
									}
									else
										this_entry.push_back(HmmState_2D(forward_pdf_class));

									while (token == "<Transition>") {
										int32 dst_state;
										BaseFloat trans_prob;
										ReadBasicType(is, binary, &dst_state);
										ReadBasicType(is, binary, &trans_prob);
										this_entry.back().transitions_left_right.push_back(std::make_pair(dst_state, trans_prob));
										ReadToken(is, binary, &token);
									}
									if (token != "</State>")
										KALDI_ERR << "Expected </State>, got instead " << token;
									ReadToken(is, binary, &token); // token should be <State> or </LeftRight> here
								}
							}
							else {
								// ���this_entry�Ѿ�����ʼ��һ�Σ����Ѿ�������<LeftRight>...</LeftRight>��ֻ��Ҫ���trans_top_down���Լ��ɣ�
								while (token != "</LeftRight>") {
									int32 state;
									ReadBasicType(is, binary, &state);
									if (state >= this_entry.size()) {
										KALDI_ERR << "Phone's pdf number for <TopDown> and <LeftRight> don't match, <LeftRight> has " << this_entry.size() << " pdfs";
									}
									ReadToken(is, binary, &token);
									int32 forward_pdf_class = kNoPdf;  // -1 by default, means no pdf.

									if (token == "<PdfClass>") {
										ReadBasicType(is, binary, &forward_pdf_class);
										if (this_entry.at(state).self_loop_pdf_class != forward_pdf_class)
											KALDI_ERR << "Phone's PdfClass don't match, TopDown's state " << state << " has PdfClass for "
											<< this_entry.at(state).self_pdf_class << " but LeftRight's state " << state << " has PdfClass for "
											<< forward_pdf_class;
										ReadToken(is, binary, &token);
										if (token == "<SelfLoopPdfClass>")
											KALDI_ERR << "pdf classes should be defined using <PdfClass> "
											<< "or <ForwardPdfClass>/<SelfLoopPdfClass> pair";
									}
									else if (token == "<ForwardPdfClass>") {
										int32 self_loop_pdf_class = kNoPdf;
										ReadBasicType(is, binary, &forward_pdf_class);
										ReadToken(is, binary, &token);
										if (token != "<SelfLoopPdfClass>")
											KALDI_ERR << "Expected <SelfLoopPdfClass>, got instead " << token;
										ReadBasicType(is, binary, &self_loop_pdf_class);
										if (this_entry.at(state).self_loop_pdf_class != self_loop_pdf_class ||
											this_entry.at(state).forward_pdf_class != forward_pdf_class)
											KALDI_ERR << "Phone's PdfClass don't match, TopDown's state " << state << " has self_loop_pdf_class for "
											<< this_entry.at(state).self_loop_pdf_class << "and forward_pdf_class for "
											<< this_entry.at(state).forward_pdf_class << " but LeftRight's state " << state
											<< " has self_loop_pdf_class for " << self_loop_pdf_class << "and forward_pdf_class for "
											<< forward_pdf_class;
										ReadToken(is, binary, &token);
									}
									else {
										if (this_entry.at(state).forward_pdf_class != forward_pdf_class)
											KALDI_ERR << "Phone's PdfClass don't match, TopDown's state " << state << " has PdfClass for "
											<< this_entry.at(state).self_pdf_class << " but LeftRight's state " << state << " has PdfClass for "
											<< forward_pdf_class;
									}

									while (token == "<Transition>") {
										int32 dst_state;
										BaseFloat trans_prob;
										ReadBasicType(is, binary, &dst_state);
										ReadBasicType(is, binary, &trans_prob);
										this_entry.at(state).transitions_left_right.push_back(std::make_pair(dst_state, trans_prob));
										ReadToken(is, binary, &token);
									}
									if (token != "</State>")
										KALDI_ERR << "Expected </State>, got instead " << token;
									ReadToken(is, binary, &token);
								}
							}
						}
						ReadToken(is, binary, &token); // ����token�ִ��д����<TopDown>��<LeftRight>��</TopologyEntry_2D>
					}
					int32 my_index = entries_.size(); // entries_��Ӧ����phone��TopologyEntry
					entries_.push_back(this_entry);
					// ��ͬһ��<TopologyEntry_2D>...</TopologyEntry_2D>�е�����phone����ͬһ��Entry����entries_��ֻ�洢һ��
					for (size_t i = 0; i < phones.size(); i++) {
						int32 phone = phones[i]; // phones��������洢��<ForPhones>�ж�ȡ�����������б�
						if (static_cast<int32>(phone2idx_.size()) <= phone)
							phone2idx_.resize(phone + 1, -1);  // -1 is invalid index. // ��-1����ʼ����Щ�²����Ԫ��
						KALDI_ASSERT(phone > 0);
						if (phone2idx_[phone] != -1)
							KALDI_ERR << "Phone with index " << (i) << " appears in multiple topology entries.";
						phone2idx_[phone] = my_index;
						phones_.push_back(phone);
					}
				}
			}
			std::sort(phones_.begin(), phones_.end());
			KALDI_ASSERT(IsSortedAndUniq(phones_));
		}
		else {  // binary I/O, just read member objects directly from disk.
		    // keep HmmTopology_2D::Read() the same order as HmmTopology_2D::Write() does in binary mode
			ReadIntegerVector(is, binary, &phones_);
			ReadIntegerVector(is, binary, &phone2idx_);
			int32 sz;
			ReadBasicType(is, binary, &sz);
			bool is_hmm = true;
			if (sz == -1) {
				is_hmm = false;
				ReadBasicType(is, binary, &sz);
			}
			entries_.resize(sz);
			for (int32 i = 0; i < sz; i++) {
				int32 thist_sz;
				ReadBasicType(is, binary, &thist_sz);
				entries_[i].resize(thist_sz);
				for (int32 j = 0; j < thist_sz; j++) {
					ReadBasicType(is, binary, &(entries_[i][j].forward_pdf_class));
					if (is_hmm)
						entries_[i][j].self_loop_pdf_class = entries_[i][j].forward_pdf_class;
					else
						ReadBasicType(is, binary, &(entries_[i][j].self_loop_pdf_class));
					int32 thiss_sz;
					ReadBasicType(is, binary, &thiss_sz);
					entries_[i][j].transitions_top_down.resize(thiss_sz);
					for (int32 k = 0; k < thiss_sz; k++) {
						ReadBasicType(is, binary, &(entries_[i][j].transitions_top_down[k].first));
						ReadBasicType(is, binary, &(entries_[i][j].transitions_top_down[k].second));
					}
					ReadBasicType(is, binary, &thiss_sz);
					entries_[i][j].transitions_left_right.resize(thiss_sz);
					for (int32 k = 0; k < thiss_sz; k++) {
						ReadBasicType(is, binary, &(entries_[i][j].transitions_left_right[k].first));
						ReadBasicType(is, binary, &(entries_[i][j].transitions_left_right[k].second));
					}
				}
			}
		}
		ExpectToken(is, binary, "</Topology_2D>");
		Check();  // Will throw if not ok.
	}

	void HmmTopology_2D::Write(std::ostream &os, bool binary) const {
		bool is_hmm = IsHmm();
		WriteToken(os, binary, "<Topology_2D>");
		if (!binary) {  // Text-mode write.
			os << "\n";
			for (int32 i = 0; i < static_cast<int32> (entries_.size()); i++) {
				WriteToken(os, binary, "<TopologyEntry_2D>");
				os << "\n";
				WriteToken(os, binary, "<ForPhones>");
				os << "\n";
				for (size_t j = 0; j < phone2idx_.size(); j++) {
					if (phone2idx_[j] == i)
						os << j << " ";
				}
				os << "\n";
				WriteToken(os, binary, "</ForPhones>");
				os << "\n";
				WriteToken(os, binary, "<TopDown>");
				os << "\n";
				for (size_t j = 0; j < entries_[i].size(); j++) {
					WriteToken(os, binary, "<State>");
					WriteBasicType(os, binary, static_cast<int32>(j));
					if (entries_[i][j].forward_pdf_class != kNoPdf) {
						if (is_hmm) {
							WriteToken(os, binary, "<PdfClass>");
							WriteBasicType(os, binary, entries_[i][j].forward_pdf_class);
						}
						else {
							WriteToken(os, binary, "<ForwardPdfClass>");
							WriteBasicType(os, binary, entries_[i][j].forward_pdf_class);
							KALDI_ASSERT(entries_[i][j].self_loop_pdf_class != kNoPdf);
							WriteToken(os, binary, "<SelfLoopPdfClass>");
							WriteBasicType(os, binary, entries_[i][j].self_loop_pdf_class);
						}
					}
					for (size_t k = 0; k < entries_[i][j].transitions_top_down.size(); k++) {
						WriteToken(os, binary, "<Transition>");
						WriteBasicType(os, binary, entries_[i][j].transitions_top_down[k].first);
						WriteBasicType(os, binary, entries_[i][j].transitions_top_down[k].second);
					}
					WriteToken(os, binary, "</State>");
					os << "\n";
				}
				WriteToken(os, binary, "</TopDown>");
				os << "\n";
				WriteToken(os, binary, "<LeftRight>");
				os << "\n";
				for (size_t j = 0; j < entries_[i].size(); j++) {
					WriteToken(os, binary, "<State>");
					WriteBasicType(os, binary, static_cast<int32>(j));
					if (entries_[i][j].forward_pdf_class != kNoPdf) {
						if (is_hmm) {
							WriteToken(os, binary, "<PdfClass>");
							WriteBasicType(os, binary, entries_[i][j].forward_pdf_class);
						}
						else {
							WriteToken(os, binary, "<ForwardPdfClass>");
							WriteBasicType(os, binary, entries_[i][j].forward_pdf_class);
							KALDI_ASSERT(entries_[i][j].self_loop_pdf_class != kNoPdf);
							WriteToken(os, binary, "<SelfLoopPdfClass>");
							WriteBasicType(os, binary, entries_[i][j].self_loop_pdf_class);
						}
					}
					for (size_t k = 0; k < entries_[i][j].transitions_left_right.size(); k++) {
						WriteToken(os, binary, "<Transition>");
						WriteBasicType(os, binary, entries_[i][j].transitions_left_right[k].first);
						WriteBasicType(os, binary, entries_[i][j].transitions_left_right[k].second);
					}
					WriteToken(os, binary, "</State>");
					os << "\n";
				}
				WriteToken(os, binary, "</LeftRight>");
				os << "\n";
				WriteToken(os, binary, "</TopologyEntry_2D>");
				os << "\n";
			}
		}
		else {
			WriteIntegerVector(os, binary, phones_); //ֱ��д��phones_����
			WriteIntegerVector(os, binary, phone2idx_); //ֱ��д��phone2idx_����
			// -1 is put here as a signal that the object has the new,
			// extended format with SelfLoopPdfClass
			if (!is_hmm) WriteBasicType(os, binary, static_cast<int32>(-1)); 
			WriteBasicType(os, binary, static_cast<int32>(entries_.size()));//д��entries_�ĸ������м��ֲ�ͬ��HMM�ṹ��
			for (size_t i = 0; i < entries_.size(); i++) {
				WriteBasicType(os, binary, static_cast<int32>(entries_[i].size()));//д��һ��entry��states����
				for (size_t j = 0; j < entries_[i].size(); j++) { //������ǰentry������states,д��pdf��trans_pair
					WriteBasicType(os, binary, entries_[i][j].forward_pdf_class);
					if (!is_hmm) WriteBasicType(os, binary, entries_[i][j].self_loop_pdf_class);
					WriteBasicType(os, binary, static_cast<int32>(entries_[i][j].transitions_top_down.size()));
					for (size_t k = 0; k < entries_[i][j].transitions_top_down.size(); k++) {
						WriteBasicType(os, binary, entries_[i][j].transitions_top_down[k].first);
						WriteBasicType(os, binary, entries_[i][j].transitions_top_down[k].second);
					}
					WriteBasicType(os, binary, static_cast<int32>(entries_[i][j].transitions_left_right.size()));
					for (size_t k = 0; k < entries_[i][j].transitions_left_right.size(); k++) {
						WriteBasicType(os, binary, entries_[i][j].transitions_left_right[k].first);
						WriteBasicType(os, binary, entries_[i][j].transitions_left_right[k].second);
					}
				}
			}
		}
		WriteToken(os, binary, "</Topology_2D>");
		if (!binary) os << "\n";
	}

	void HmmTopology_2D::Check() {
		if (entries_.empty() || phones_.empty() || phone2idx_.empty())
			KALDI_ERR << "HmmTopology_2D::Check(), empty object.";
		std::vector<bool> is_seen(entries_.size(), false);
		for (size_t i = 0; i < phones_.size(); i++) {
			int32 phone = phones_[i];
			if (static_cast<size_t>(phone) >= phone2idx_.size() ||
				static_cast<size_t>(phone2idx_[phone]) >= entries_.size())
				KALDI_ERR << "HmmTopology_2D::Check(), phone has no valid index.";
			is_seen[phone2idx_[phone]] = true;
		}
		for (size_t i = 0; i < entries_.size(); i++) {
			if (!is_seen[i])
				KALDI_ERR << "HmmTopology_2D::Check(), entry with no corresponding phones.";
			int32 num_states = static_cast<int32>(entries_[i].size());
			if (num_states <= 1)
				KALDI_ERR << "HmmTopology_2D::Check(), cannot only have one state (i.e., must "
				"have at least one emitting state).";
			if ((!entries_[i][num_states - 1].transitions_left_right.empty())||
				(!entries_[i][num_states - 1].transitions_top_down.empty()))
				KALDI_ERR << "HmmTopology_2D::Check(), last state must have no transitions.";
			// not sure how necessary this next stipulation is.
			//if (entries_[i][num_states - 1].forward_pdf_class != kNoPdf)
				//KALDI_ERR << "HmmTopology_2D::Check(), last state must not be emitting.";
			/*
			std::vector<bool> has_trans_in(num_states, false);
			std::vector<int32> seen_pdf_classes;
			
			for (int32 j = 0; j < num_states; j++) {  // j is the state-id.
				BaseFloat tot_prob = 0.0;
				if (entries_[i][j].forward_pdf_class != kNoPdf) {
					seen_pdf_classes.push_back(entries_[i][j].forward_pdf_class);
					seen_pdf_classes.push_back(entries_[i][j].self_loop_pdf_class);
				}
				std::set<int32> seen_transition;
				for (int32 k = 0;
					static_cast<size_t>(k) < entries_[i][j].transitions.size();
					k++) {
					
					tot_prob += entries_[i][j].transitions[k].second;
					if (entries_[i][j].transitions[k].second <= 0.0)
						KALDI_ERR << "HmmTopology_2D::Check(), negative or zero transition prob.";
					int32 dst_state = entries_[i][j].transitions[k].first;
					// The commented code in the next few lines disallows a completely
					// skippable phone, as this would cause to stop working some mechanisms
					// that are being built, which enable the creation of phone-level lattices
					// and rescoring these with a different lexicon and LM.
					if (dst_state == num_states - 1 // && j != 0
						&& entries_[i][j].forward_pdf_class == kNoPdf)
						KALDI_ERR << "We do not allow any state to be "
						"nonemitting and have a transition to the final-state (this would "
						"stop the SplitToPhones function from identifying the last state "
						"of a phone.";
					
					if (dst_state < 0 || dst_state >= num_states)
						KALDI_ERR << "HmmTopology_2D::Check(), invalid dest state " << (dst_state);
					if (seen_transition.count(dst_state) != 0)
						KALDI_ERR << "HmmTopology_2D::Check(), duplicate transition found.";
					if (dst_state == k) {  // self_loop...
						KALDI_ASSERT(entries_[i][j].self_pdf_class != kNoPdf &&
							"Nonemitting states cannot have self-loops.");
					}
					seen_transition.insert(dst_state);
					has_trans_in[dst_state] = true;
				}
				if (j + 1 < num_states) {
					KALDI_ASSERT(tot_prob > 0.0 && "Non-final state must have transitions out."
						"(with nonzero probability)");
					if (fabs(tot_prob - 1.0) > 0.01)
						KALDI_WARN << "Total probability for state " << j <<
						" in topology entry is " << tot_prob;
				}
				else
					KALDI_ASSERT(tot_prob == 0.0);
			}
			
			// make sure all but start state have input transitions.
			for (int32 j = 1; j < num_states; j++)
				if (!has_trans_in[j])
					KALDI_ERR << "HmmTopology_2D::Check, state " << (j) << " has no input transitions.";
			SortAndUniq(&seen_pdf_classes);
			if (seen_pdf_classes.front() != 0 ||
				seen_pdf_classes.back() != static_cast<int32>(seen_pdf_classes.size()) - 1) {
				KALDI_ERR << "HmmTopology_2D::Check(), pdf_classes are expected to be "
					"contiguous and start from zero.";
			}
			*/
		}
	}

	//return true if forward_pdf_class=self_loop_pdf_class
	bool HmmTopology_2D::IsHmm() const {
		const std::vector<int32> &phones = GetPhones();
		KALDI_ASSERT(!phones.empty());
		for (size_t i = 0; i < phones.size(); i++) {
			int32 phone = phones[i];
			const TopologyEntry_2D &entry = TopologyForPhone(phone);
			for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
				int32 forward_pdf_class = entry[j].forward_pdf_class,
					self_loop_pdf_class = entry[j].self_loop_pdf_class;
				if (forward_pdf_class != self_loop_pdf_class)
					return false;
			}
		}
		return true;
	}

	const HmmTopology_2D::TopologyEntry_2D& HmmTopology_2D::TopologyForPhone(int32 phone) const {  // Will throw if phone not covered.
		if (static_cast<size_t>(phone) >= phone2idx_.size() || phone2idx_[phone] == -1) {
			KALDI_ERR << "TopologyForPhone(), phone " << (phone) << " not covered.";
		}
		return entries_[phone2idx_[phone]];
	}

	int32 HmmTopology_2D::NumPdfClasses(int32 phone) const {
		// will throw if phone not covered.
		const TopologyEntry_2D &entry = TopologyForPhone(phone);
		int32 max_pdf_class = 0;
		for (size_t i = 0; i < entry.size(); i++) {
			max_pdf_class = std::max(max_pdf_class, entry[i].self_pdf_class);
		}
		return max_pdf_class + 1;
	}

	// TODO, this function has not been checked yet! Seems no use.
	int32 HmmTopology_2D::MinLength(int32 phone) const {
		const TopologyEntry_2D &entry = TopologyForPhone(phone);
		// min_length[state] gives the minimum length for sequences up to and
		// including that state.
		std::vector<int32> min_length(entry.size(),
			std::numeric_limits<int32>::max());
		KALDI_ASSERT(!entry.empty());

		min_length[0] = (entry[0].self_pdf_class == -1 ? 0 : 1);
		int32 num_states = min_length.size();
		bool changed = true;
		/*
		while (changed) {
			changed = false;
			for (int32 s = 0; s < num_states; s++) {
				const HmmState &this_state = entry[s];
				std::vector<std::pair<int32, BaseFloat> >::const_iterator
					iter = this_state.transitions.begin(),
					end = this_state.transitions.end();
				for (; iter != end; ++iter) {
					int32 next_state = iter->first;
					KALDI_ASSERT(next_state < num_states);
					int32 next_state_min_length = min_length[s] +
						(entry[next_state].forward_pdf_class == -1 ? 0 : 1);
					if (next_state_min_length < min_length[next_state]) {
						min_length[next_state] = next_state_min_length;
						if (next_state < s)
							changed = true;
						// the test of 'next_state < s' is an optimization for speed.
					}
				}
			}
		}
		
		KALDI_ASSERT(min_length.back() != std::numeric_limits<int32>::max());
		// the last state is the final-state.
		return min_length.back();
		*/
		return 0;
	}

} // End namespace kaldi
